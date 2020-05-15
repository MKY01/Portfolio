

from mrjob.job import MRJob
from mrjob.step import MRStep
import re

WORD_RE = re.compile(r"[\w']+") # from https://docs.python.org/3/library/re.html#module-re

class MRBigramWordsFreq(MRJob):

    SORT_VALUES = True

    def steps(self):
        return[
            MRStep(mapper=self.mapper_find_words,
                    combiner=self.combiner_combine_counts,
                    reducer=self.reducer_sum_counts),
            MRStep(reducer=self.reducer_compute_freq)
        ]

    def mapper_find_words(self, _, line):
        prev_word = None

        for word in WORD_RE.findall(line):
            word = word.lower()

            if prev_word != None:
                yield (prev_word, '*'), 1
                yield (prev_word, word), 1

            prev_word = word

            
    def combiner_combine_counts(self, key, counts):
        yield key, sum(counts)

        
    def reducer_sum_counts(self, key, counts):

        count = sum(counts)
        prev_word, word = key

        if word == '*':
            yield prev_word, ('01: total', count)
        else:
            yield prev_word, ('02: stats', (word, count))

            
    def reducer_compute_freq(self, prev_word, value):
        total = None

        for value_type, data in value:
            if value_type == '01: total':
                total = data
            else:
                assert value_type == '02: stats'
                word, count = data
                percent = 100.0 * count / total
                yield (prev_word, word), (total, count, percent)

                
if __name__ == '__main__':
    MRBigramWordsFreq.run()
