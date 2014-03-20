from datetime import datetime
from glob import glob

__author__ = 'tamponi'

if __name__ == "__main__":
    today_prefix = "results_{:%Y%m%d}".format(datetime.utcnow())

    next_dir = "results/{}_{:02d}".format(today_prefix, len(glob("./"+today_prefix+"*"))+1)
    print next_dir
