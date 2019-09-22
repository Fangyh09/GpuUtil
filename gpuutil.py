# only one gpu
# import subprocess
import fire
import re
import GPUtil
import sys
from io import StringIO  # Python3
# from loguru import logger
import subprocess
# from multiproc
from concurrent.futures import ThreadPoolExecutor as Pool
# from utility import *

def replace_n(string):
    regex = re.compile(r'[\\n]')
    string = regex.sub('\n', string)
    return string


def get_io(fun):
    old_stdout = sys.stdout
 
    # This variable will store everything that is sent to the standard output
    
    result = StringIO()
    
    sys.stdout = result
    
    # Here we can call anything we like, like external modules, and everything that they will send to standard output will be stored on "result"
    
    fun()
    
    # Redirect again the std output to screen
    
    sys.stdout = old_stdout
    
    # Then, get the stdout like a string and process it!
    
    result_string = result.getvalue()
    return result_string


def gpustatus_wrapper(gpuids, outfile):
    # gpuids = gpuids.split(",")
    if len(str(gpuids)) == 1:
        gpuids = [int(gpuids)]
    else:
        # gpuids = gpuids.split(",")
        gpuids = [int(x) for x in gpuids]
    return gpustatus(gpuids, outfile)

def gpustatus(gpuids=[0, 1, 2, 3, 4, 5, 6, 7], outfile=None):
    """
    return nvidia-smi info && gpustatus info
    """
    # result = subprocess.run(['bash', './gpustatus.sh'], stdout=subprocess.PIPE)
    # result = subprocess.run(['bash', './gpustatus.sh'])
    # result = subprocess.run(['nvidia-smi'])
    attrList = [[{'attr':'id','name':'ID'},
                        {'attr':'name','name':'Name'}],
                    [
                        {'attr':'load','name':'GPU util.','suffix':'%','transform': lambda x: x*100,'precision':0},
                        {'attr':'memoryUtil','name':'Memory util.','suffix':'%','transform': lambda x: x*100,'precision':0}],
                    [{'attr':'memoryUsed','name':'Memory used','suffix':'MB','precision':0},
                        {'attr':'memoryTotal','name':'Memory total','suffix':'MB','precision':0}]]
    def fun():
        GPUtil.showUtilization(all=False, attrList=attrList, gpuids=gpuids)
    output = get_io(fun)
    # output = subprocess.check_output('nvidia-smi', shell=True)
    # output = str(output)
    # output = replace_n(output)
    # print(output)
    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(output)
    return output


def fetch_empty_gpu():
    """
    return empty gpus id
    """
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 10, maxLoad = 0.01, maxMemory = 0.01, excludeID=[], excludeUUID=[])
    return deviceIDs

def get_free_gpustatus():
    deviceIDs = fetch_empty_gpu()
    output = gpustatus(gpuids=deviceIDs)
    return output

def get_one_gpu():
    """
    return: is a list
    """
    deviceIDs = fetch_empty_gpu()
    return deviceIDs


if __name__ == '__main__':
    fire.Fire()
    # output = gpustatus()
    # logger.debug(output)
