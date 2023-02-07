#!/usr/bin/env python3.9
import sys
import time
import resource
import json
import os
from collections import deque
import yaml
import pathlib
import itertools
import multiprocessing
from pathlib import Path
import collections
import subprocess
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import random


LOG = logging.getLogger(__name__)


def _gen_rand_byte(blocksize, repeat=1):
    while True:
        data = random.randbytes(blocksize)
        loop = repeat
        while loop > 0:
            yield data
            loop -= 1


def _gen_file(conf, filepath: pathlib.Path, seed: str):
    blocksize = conf.blocksize
    random.seed(seed)
    rand_size = random.randint(0, conf.delta)
    # 增加一个 1G 左右的变化
    size = conf.size + rand_size * blocksize
    if filepath.exists():
        if filepath.stat().st_size == size:
            LOG.info("%s exists and in the exptected size: %s, skip it",
                     filepath, size)
            return
        else:
            LOG.warning("%s exists with size: %s, not in the expected: %s",
                        filepath, filepath.stat().st_size,
                        size)
    LOG.info("Generate %s with filesize: %s seed: %s", filepath, size, seed)
    with open(filepath, 'wb') as f:
        for data in _gen_rand_byte(blocksize, 1):
            f.write(data)
            size -= len(data)
            if size <= 0:
                break
    return filepath


CarFile = collections.namedtuple('CarFile', ['index', 'fullpath', 'filename', 'payload_cid', 'commp_cid', 'piece_size', 'car_size', 'seed'])

def zip_p():
    pass

def process(conf, file, output_queue):
    if conf.dry_run:
        return
    output = output_queue.get()

    index = file['index']
    seed = file['seed']
    filename = '%05d' % index
    src_file = pathlib.Path(conf.temp, filename)
    dest_file = pathlib.Path(output, '%s.car' % filename)
    output_path = dest_file.parent
    state_folder = dest_file.parent / 'state'
    state_file = state_folder / dest_file.name
    #if not output_path.exists():
    if not os.path.exists(str(output_path)):
        os.mkdir(output_path)
    #if not state_folder.exists():
    if not os.path.exists(str(state_folder)):
        print('报错报错99999999999999999')
        os.mkdir(state_folder)
    start = time.time()
    try:
        _gen_file(conf, src_file, seed)
        LOG.info("Raw file %s is generated and start generate the car file, took: %d",
                 filename, int(time.time() - start))
        start = time.time()
        #加密代码
        cmd_zip = "zip -P '14z^psC^iTzPUEs#cZdMuMWabVY&&NKE@IEoJGXl152WpFhcFuRDjCM&Aq5VZFS*&' -0 {}.zip {}".format(src_file,src_file)
        if_zip = os.system(cmd_zip)
        print(if_zip)
        zip_path = "{}.zip".format(src_file)
        zip_path = pathlib.Path(zip_path)
        if zip_path.exists():
            print("压缩成功{}.zip".format(src_file))
        cmd = [conf.boostx_binary, 'generate-car', "{}.zip".format(src_file), dest_file]
        print(cmd)
        env = os.environ.copy()
        env['TMPDIR'] = '/dev/shm'
        stdout = subprocess.check_output(
            cmd, stderr=subprocess.PIPE, env=env).decode('utf8')
        LOG.info("car file for %s is generate in %s and start generate the commp, took: %s",
                 filename, dest_file, int(time.time() - start))
        payload_cid = yaml.safe_load(stdout)['Payload CID']

        cmd = [conf.boostx_binary, 'commp', dest_file]
        print(cmd)
        stdout = subprocess.check_output(
            cmd, stderr=subprocess.PIPE, env=env).decode('utf8')
        data = yaml.safe_load(stdout)
        commp_cid = data['CommP CID']
        piece_size = data['Piece size']
        car_size = data['Car file size']
        carfile = CarFile(index, dest_file, filename, payload_cid,
                          commp_cid, piece_size, car_size, seed)
        # save the state file
        with open(state_file, 'w') as f:
            data = {
                "Payload CID": payload_cid,
                "CommP CID": commp_cid,
                "Piece size": piece_size,
                "Car file size": car_size}
            f.write(json.dumps(data, indent=2))
        return carfile
    except Exception:
        if dest_file.exists():
            os.remove(dest_file)
        raise
    finally:
        if not conf.keep_src and src_file.exists():
            os.remove(src_file)
            os.remove("{}.zip".format(src_file))
        output_queue.put(output)


def save_config(conf, config):
    data = json.dumps(config, indent=2)
    with open(conf.config, 'w') as f:
        f.write(data)


def _config_logging(conf):
    level = logging.INFO
    if conf.debug:
        level = logging.DEBUG

    path = f'{conf.config}.log'

    logging.basicConfig(
        level=level,
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(process)d %(levelname)05s  %(message)s")
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.INFO)


class TimeTrack:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start = time.time()
        res = self.func(*args, **kwargs)
        elapsed = time.time() - start
        return elapsed, res


class TimeLeft:

    def __init__(self, maxlen=50):
        self.queue = deque(maxlen=maxlen)

    def add_elapsed(self, elapsed: float):
        self.queue.appendleft(elapsed)

    def average(self):
        if len(self.queue) == 0:
            return 0
        return sum(self.queue) / len(self.queue)

    def estimate(self, count: int):
        return int(self.average()) * count


def human_seconds(second):
    '''
    >>> human_seconds(100)
    '00:01:40'
    >>> human_seconds(1)
    '00:00:01'
    >>> human_seconds(99999)
    '27:46:39'
    '''
    x = y = z = 0
    y, z = divmod(int(second), 60)
    if y > 60:
        x, y = divmod(y, 60)
    return f'{x:0>2}:{y:0>2}:{z:0>2}'


def _fix_ulimit_nofile():
    target = 1024000
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if target > hard:
        raise ValueError("Can not set nofile to %d, hard is %d" %
                         (target, hard))
    if target >= soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    LOG.info("nofile is %s", resource.getrlimit(resource.RLIMIT_NOFILE))


def main():
    pyversion = sys.version_info
    if pyversion.major != 3 or pyversion.minor != 9:
        raise ValueError("Need python 3.9, got %s" % sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--boostx-binary', default='boostx')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Dry run mode')
    parser.add_argument('--config-only', action='store_true',
                        help='generate config only')
    parser.add_argument('--temp', type=str,
                        default='/dev/shm/boostx')
    parser.add_argument('-o', '--output', required=True, type=str,
                        nargs='+', help='Target folder')
    parser.add_argument('--blocksize', type=int, default=4*1024*1024,
                        help='Block size, defalt is 4M')
    parser.add_argument('--size', type=int, default=18*1024**3,
                        help='File size in bytes, defualt is 18G')
    parser.add_argument('--delta', type=int, default=100,
                        help='在现在文件大小上增加的增量，增加大小为 blocksize * delta')
    parser.add_argument('--concurrent', '-c', type=int, default=24)
    parser.add_argument('--keep-src', action='store_true')
    parser.add_argument('--total', type=int)
    parser.add_argument('--debug', '-d',  action='store_true')
    parser.add_argument('--seed', type=str)
    parser.add_argument('--config', type=str, required=True)
    conf = parser.parse_args()
    _config_logging(conf)
    _fix_ulimit_nofile()

    temp_folder = Path(conf.temp)
    if not temp_folder.exists():
        os.makedirs(temp_folder)

    with open(conf.config) as f:
        config = json.load(f)

    if conf.total:
        config['total'] = conf.total
        total = conf.total
    else:
        total = config['total']
    if conf.seed:
        config['seed'] = conf.seed

    config['output'] = str(pathlib.Path(conf.output[0]).absolute())
    files = config.get('files', [])
    file_idx = set([file['index'] for file in files])
    for n in range(int(total)):
        idx = n + 1
        if idx in file_idx:
            continue
        files.append({
            'index': idx,
            'seed': f'{config["seed"]}:{idx}',
            'status': 'new'
        })
    config['files'] = files
    save_config(conf, config)

    LOG.info("Config file is updated")
    if conf.config_only:
        return

    output_queue = multiprocessing.Manager().Queue()
    for _, output in zip(range(conf.concurrent), itertools.cycle(conf.output)):
        output_queue.put(output)

    exector = ProcessPoolExecutor(max_workers=conf.concurrent)

    tasks = []

    for file in config['files']:
        filename = '%s.car' % file.get('filename', '__not_exist_file__')
        #if file['status'] == 'done' and (Path(config['output']) / filename).exists():
        if file['status'] == 'done':
            LOG.info("skip index: %d with filename %s is done already",
                     file['index'], file['filename'])
            continue
        else:
            LOG.debug("Need genearte file %s", file)
        tasks.append(exector.submit(
            TimeTrack(process), conf, file, output_queue))

    total = len(tasks)
    done = failed = 0
    time_left = TimeLeft()
    try:
        for task in as_completed(tasks):
            try:
                elapsed, carfile = task.result()
                if conf.dry_run:
                    continue
                done += 1
                time_left.add_elapsed(elapsed)
                idx = carfile.index
                idx -= 1
                files[idx]['status'] = 'done'
                files[idx]['fullpath'] = str(carfile.fullpath)
                files[idx]['filename'] = carfile.filename
                files[idx]['payloadCid'] = carfile.payload_cid
                files[idx]['commpCid'] = carfile.commp_cid
                files[idx]['pieceSize'] = carfile.piece_size
                files[idx]['carSize'] = carfile.car_size
                files[idx]['seed'] = carfile.seed
                output = config['output']
                print("正在进行重命名：{} ==> {}/{}.car".format(str(carfile.fullpath),output,carfile.commp_cid))
                print("输出目录：",output)
                os.rename("{}".format(str(carfile.fullpath)),"{}/{}.car".format(output,carfile.commp_cid))
                task_left = total - done - failed
                save_config(conf, config)
                LOG.info("file rename  Success {} ==> {}/{}.car".format(str(carfile.fullpath),output,carfile.commp_cid))
                LOG.info("Success %d/%d/%d took: %s, eta: %s, %s",
                         done, failed, total,
                         human_seconds(elapsed),
                         human_seconds(time_left.average() * task_left/conf.concurrent),
                         carfile.fullpath)
            except Exception as ex:
                failed += 1
                task_left = total - done - failed
                LOG.exception("Success %d/%d/%d eta: %s: %s",
                              done, failed, total,
                              human_seconds(time_left.average() * task_left/conf.concurrent),
                              str(ex))
    finally:
        save_config(conf, config)


if __name__ == "__main__":
    main()
