'''
This is how I start 99% of my python scripts
'''
import argparse
import logging
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('example')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    # Do things here
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')