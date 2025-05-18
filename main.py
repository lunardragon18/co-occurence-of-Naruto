import os
import sys

from ner.ner import NER
from character_network.character_network import Networks

def main():
    ner = NER()
    ner = ner.get_ner("Data/Subtitles","output/output.csv")
    network = Networks()
    ner_df = Networks.ner(ner)
    network.draw_network(ner_df)

if __name__ == "__main__":
    main()
