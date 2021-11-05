# Prepare file lists and save in the 'filelists' directory
import argparse
import os
import glob
import random
import wave
import contextlib


def read_meta_ljspeech(metafile):
  """read ljspeech metadata.csv to get filename to text mapping"""

  lines = open(metafile).readlines()
  filename2text = {}
  for i, line in enumerate(lines):
    filename, text, text_normed = line.rstrip().split('|')
    basename = os.path.splitext(os.path.basename(filename))[0]
    filename2text[basename] = text_normed
  return filename2text


def prep_ljspeech(args, verbose=False):
  """prepare lines in filelist for ljspeech"""

  wavfiles = glob.glob(os.path.join(args.input_dir, '*.wav'))
  wavfiles = sorted(wavfiles, key=lambda f: os.path.basename(f))
  nwavfiles = len(wavfiles)
  if verbose:
    print('# of wavs in {}: {}'.format(args.input_dir, nwavfiles))

  filename2text = read_meta_ljspeech(args.metafile)

  filename2dur = {}
  batchsize = 1000
  for i, wavfile in enumerate(wavfiles):
    if verbose and (i%batchsize == 0):
      print('getting duration for file {} ~ {}'.format(
        i, min(i+batchsize, nwavfiles)))
    basename = os.path.splitext(os.path.basename(wavfile))[0]
    filename2dur[basename] = wav_duration(wavfile)

  lines = ['' for _ in range(len(wavfiles))]
  for i, wavfile in enumerate(wavfiles):
    basename = os.path.splitext(os.path.basename(wavfile))[0]
    text, dur = filename2text[basename], filename2dur[basename]
    lines[i] = '|'.join([wavfile, text, str(dur)])

  return lines


def split(lines, ratio='8:1:1', seed=0, ordered=True):
  """split lines by ratios for train/valid/test"""

  # get line indecies to split
  ratios = [float(r) for r in ratio.split(':')]
  percents = [sum(ratios[:i+1]) for i in range(len(ratios))]
  percents = [p/sum(ratios) for p in percents]
  nlines = len(lines)
  idxs = [0] + [int(p*nlines) for p in percents]

  # shuffle lines with fixed random seed
  random.seed(seed)
  random.shuffle(lines)

  cats = ['train', 'valid', 'test']
  if ordered:
    flist = {cat:sorted(lines[idxs[i]:idxs[i + 1]]) for (cat,i)
             in zip(cats, range(len(cats)))}
  else:
    flist = {cat:lines[idxs[i]:idxs[i + 1]] for (cat,i)
             in zip(cats, range(len(cats)))}
  return flist


def wav_duration(filename):
  """get wav file duration in seconds"""
  with contextlib.closing(wave.open(filename,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    #print(duration)
  return duration


def sort_by_dur(lines, reverse=True):
  """sort lines by duration (as one item in the line)"""
  lines_sorted = sorted(lines, key=lambda line:float(line.split('|')[-3]),
                        reverse=reverse)
  return lines_sorted


def write_flist(flist, dataset, output_dir, verbose=True):

  # update output directory to include dataset name
  output_dir = os.path.join(output_dir, dataset)
  os.makedirs(output_dir, exist_ok=True)

  for cat in flist.keys():
    listfile = '{}_wav_{}.txt'.format(dataset, cat)
    listpath = os.path.join(output_dir, listfile)
    open(listpath, 'w').write('\n'.join(flist[cat]))
    if verbose:
      print('wrote list file: {}'.format(listpath))


def parse_args():

  usage = 'usage: prepare file lists'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-i', '--input-dir', type=str,
                      help='directory for input data')
  parser.add_argument('-o', '--output-dir',type=str,
                      help='directory for output file lists')
  parser.add_argument('-d', '--dataset', required=True,
                      choices=['ljspeech'])
  parser.add_argument('-r', '--ratio', type=str, default='8:1:1',
                      help='train/valid/test ratios')
  parser.add_argument('--metafile', type=str, default='',
                      help='optional metadata file')
  parser.add_argument('--seed', type=int, default=0,
                      help=('random seed to split filelist'
                            ' into training, validation and test'))
  parser.add_argument('--ordered', action='store_true',
                      help=('file list will be sorted after randomization'
                            ' if specified'))

  return parser.parse_args()


def main():

  args = parse_args()

  # # example
  # args = argparse.ArgumentParser()
  # args.ratio = '8:1:1'
  # args.seed = 0
  # args.ordered = False
  # args.output_dir = 'filelists'
  #
  # # ljspeech
  # args.input_dir = r'data/LJSpeech-1.1/wavs'
  # args.dataset = 'ljspeech'
  # args.metafile = r'data/LJSpeech-1.1/metadata.csv'

  if args.dataset == 'ljspeech':
    lines = prep_ljspeech(args)
  else:
    raise Exception('{} is not supported'.format(args.dataset))

  # split lines into train/valid/test
  flist = split(lines, ratio=args.ratio, seed=args.seed, ordered=args.ordered)

  # sort each set by duration (for efficient batching)
  for cat in flist.keys():
    flist[cat] = sort_by_dur(flist[cat], reverse=True)

  # write out file lists
  write_flist(flist, args.dataset, args.output_dir)

if __name__ == "__main__":
  main()
