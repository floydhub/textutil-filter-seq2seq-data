"""
Preprocess text
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import codecs
import csv
import spacy

def str2bool(val):
    """
    Helper method to convert string to bool
    """
    if val is None:
        return False
    val = val.lower().strip()
    if val in ['true', 't', 'yes', 'y', '1', 'on']:
        return True
    elif val in ['false', 'f', 'no', 'n', '0', 'off']:
        return False

def filter_sentences(sentences, max_words=25, src=False):
    """
    Filter sentences to satisfy maxWords.
    If src, keep the last sentences
    If tgt, keep only the first sentences
    """
    if src:
        sentences = reversed(list(sentences))

    lines = []
    word_count = 0
    for sentence in sentences:
        num_words = len(sentence)
        if word_count == 0 or word_count + num_words <= max_words:
            lines.append(str(sentence))
            word_count += num_words
        else:
            break

    if src:
        lines = reversed(lines)

    return ' '.join(lines)

def main():
    """
    Filters Seq2Seq data to the given word limits for source and target sentences.
    Basically, this attempts to retain the last few sentences (within word limits)
    from the src, and the first sentence from the tgt.
    """

    # Parse command line args
    parser = argparse.ArgumentParser(description='Filter Seq2Seq data')

    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to input file')
    parser.add_argument(
        '-d', '--delimiter', required=True, default='\t', 
        help='Column delimiter between columns')
    parser.add_argument(
        '-l', '--language', required=True, default='en', choices=['en', 'de'],
        help='Language code to load Spacy language model')
    parser.add_argument(
        '-m', '--maxWords', required=True, default=25, type=int,
        help='Maximum number of words in a src/tgt sentence')
    parser.add_argument(
        '-header', '--hasheader', required=False, type=str2bool,
        default='False', help='File has header row?')
    parser.add_argument('-o', '--output', required=True, help='Path to output file')

    # Text preprocess args
    parser.add_argument(
        '--fix_unicode', required=False, type=str2bool,
        default='False', help='if True, fix “broken” unicode such as mojibake and garbled HTML entities')
    
    args = parser.parse_args()
    # Unescape the delimiter
    args.delimiter = codecs.decode(args.delimiter, "unicode_escape")

    # Convert args to dict
    vargs = vars(args)

    print("\nArguments:")
    for arg in vargs:
        print("{}={}".format(arg, getattr(args, arg)))

    # Load SpaCy
    nlp = spacy.load(args.language)

    # Read the input file
    with open(args.input, 'r') as inputfile:
        with open(args.output, 'w') as outputfile:
            
            reader = csv.reader(inputfile, delimiter=args.delimiter)
            writer = csv.writer(outputfile, delimiter=args.delimiter)

            # If has header, write it unprocessed
            if args.hasheader:
                headers = next(reader, None)
                if headers:
                    writer.writerow(headers)

            print("\nProcessing input")
            for row in reader:
                if not len(row) == 2:
                    raise AssertionError("Expected 2 cols (src, tgt). Received: {}".format(
                        len(row)
                    ))

                src = row[0]
                src_doc = nlp(src)
                src_filt = filter_sentences(sentences=src_doc.sents, max_words=args.maxWords, src=True)

                tgt = row[1]
                tgt_doc = nlp(tgt)
                tgt_filt = filter_sentences(sentences=tgt_doc.sents, max_words=args.maxWords, src=False)

                writer.writerow([src_filt, tgt_filt])

    print("\nDone. Bye!")

if __name__ == '__main__':
    main()

