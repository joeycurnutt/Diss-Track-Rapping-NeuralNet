# Code partially used from Robbie Barrat's Rapping NN
import pronouncing
import markovify
import re
import random
import numpy as np
import os
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
import tensorflow as tf
from profanity_filter import ProfanityFilter
pf = ProfanityFilter()
depth = 6 # At least 1
maxsyllables = 14
train_mode = True
artist = "Eminem"
rap_file = "diss_track.txt"
clean_mode = False

def sort_list(list):
    list.sort()
    return list

def purify(elements):
    for ele in elements:
        clean = []
        ele = re.sub(r" ?\([^)]+\)", "", ele)
        clean.append(ele)
    return clean

def create_NN(depth, is_NN):
    count = 1
    model = Sequential()
    model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))
    model.add(Dropout(0.2))
    while 4 + count <= depth:
        model.add(LSTM(4 + count, return_sequences=True))
        model.add(Dropout(0.2))
        count += 1
    count = 0
    while depth - count > 2:
        model.add(LSTM(depth - count, return_sequences=True))
        model.add(Dropout(.2))
        count += 1
    model.add(Dropout(0.2))
    if is_NN == True:
        model.add(Dense(2, activation='relu'))
    else:
        model.add(Dense(2, activation='tanh'))
    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-07,
                                       amsgrad=False,
                                       name='Adam'
                                       ),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

    if artist + ".rap" in os.listdir(".") and train_mode == False:
        model.load_weights(str(artist + ".rap"))
        print("loading saved network: " + str(artist) + ".rap")
    return model


def markov(text_file):
    read = open(text_file, "r").read()
    text_model = markovify.NewlineText(read)
    return text_model

def syllables(line):
    count = 0
    for word in line.split(" "):
        vowels = 'aeiouy'
        word = word.lower().strip("-.:;?!'")
        try:
            if word[0] in vowels:
                count += 1
        except:
            pass
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
    return count / maxsyllables

def rhymeindex(lyrics):
    if str(artist) + ".rhymes" in os.listdir(".") and train_mode == False:
        print("loading saved rhymes from " + str(artist) + ".rhymes")
        return open(str(artist) + ".rhymes", "r").read().split("\n")
    else:
        rhyme_master_list = []
        print("Alright, building the list of all the rhymes")
        for i in lyrics:
            if i is None:
                continue
            unclean = []
            unclean.append(i)
            pure_bar = purify(unclean)
            bar_as_string = pure_bar[0]
            word = bar_as_string[-2:]
            rhymeslist = pronouncing.rhymes(word)
            rhymeslistends = []
            for i in rhymeslist:
                i.strip(";:.,/?!-")
                rhymeslistends.append(i[-2:])
            try:
                rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
            except Exception:
                word.strip("'?.!-:;").lower()
                rhymescheme = word[-2:]

            rhyme_master_list.append(rhymescheme)

        rhyme_master_list = list(set(rhyme_master_list))
        reverselist = [x[::-1] for x in rhyme_master_list]
        sort_list(reverselist)
        rhymelist = [x[::-1] for x in reverselist]
        f = open(str(artist) + ".rhymes", "w")
        f.write("\n".join(rhymelist))
        f.close()
        print(rhymelist)
        return rhymelist

def rhyme(line, rhyme_list):
    word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
    rhymeslist = pronouncing.rhymes(word)
    rhymeslistends = []
    for i in rhymeslist:
        rhymeslistends.append(i[-2:])
    try:
        rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
    except Exception:
        rhymescheme = word[-2:]
    try:
        float_rhyme = rhyme_list.index(rhymescheme)
        float_rhyme = float_rhyme / float(len(rhyme_list))
        return float_rhyme
    except Exception:
        return None

def split_lyrics_file(text_file):
    text = open(text_file).read()
    text = text.split("\n")
    while "" in text:
        text.remove("")
    return text

def generate_lyrics(lyrics_file):
    bars = []
    last_words = []
    lyriclength = len(open(lyrics_file).read().split("\n"))
    count = 0
    markov_model = markov(lyrics_file)

    while len(bars) < lyriclength / 9 and count < lyriclength * 2:
        bar = markov_model.make_sentence()
        if type(bar) != type(None) and syllables(bar) < 1:
            def get_last_word(bar):
                last_word = bar.split(" ")[-1]
                if last_word[-1] in "!.?,":
                    last_word = last_word[:-1]
                return last_word
            last_word = get_last_word(bar)
            if bar not in bars and last_words.count(last_word) < 3:
                bars.append(bar)
                last_words.append(last_word)
                count += 1
    return bars
def build_dataset(lyrics, rhyme_list):
    dataset = []
    line_list = []
    for line in lyrics:
        line_list = [line, syllables(line), rhyme(line, rhyme_list)]
        dataset.append(line_list)

    x_data = []
    y_data = []

    for i in range(len(dataset) - 3):
        line1 = dataset[i][1:]
        line2 = dataset[i + 1][1:]
        line3 = dataset[i + 2][1:]
        line4 = dataset[i + 3][1:]

        x = [line1[0], line1[1], line2[0], line2[1]]
        x = np.array(x)
        x = x.reshape(2, 2)
        x_data.append(x)

        y = [line3[0], line3[1], line4[0], line4[1]]
        y = np.array(y)
        y = y.reshape(2, 2)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print(x_data)
    print(y_data)

    return x_data, y_data

def compose_rap(lines, rhyme_list, lyrics_file, model):
    rap_vectors = []
    human_lyrics = split_lyrics_file(lyrics_file)
    initial_index = random.choice(range(len(human_lyrics) - 1))
    initial_lines = human_lyrics[initial_index:initial_index + 8]
    starting_input = []
    for line in initial_lines:
        starting_input.append([syllables(line), rhyme(line, rhyme_list)])
    starting_vectors = model.predict(np.array([starting_input]).astype(float).flatten().reshape(4, 2, 2))
    rap_vectors.append(starting_vectors)

    for i in range(49):
        rap_vectors.append(model.predict(np.array([rap_vectors[-1]]).astype(float).flatten().reshape(4, 2, 2)))
    return rap_vectors

def vectors_into_song(vectors, generated_lyrics, rhyme_list):
    print("\n\n")
    print("About to write rap (this could take a moment)...")
    print("\n\n")

    def last_word_compare(rap, line2):
        penalty = 0
        for line1 in rap:
            word1 = line1.split(" ")[-1]
            word2 = line2.split(" ")[-1]

            while word1[-1] in "?!,. ":
                word1 = word1[:-1]

            while word2[-1] in "?!,. ":
                word2 = word2[:-1]

            if word1 == word2:
                penalty += 0.2

        return penalty
    def calculate_score(vector_half, syllables, rhyme, penalty):
        desired_syllables = vector_half[0]
        desired_rhyme = vector_half[1]
        desired_syllables = desired_syllables * maxsyllables
        desired_rhyme = desired_rhyme * len(rhyme_list)
        try:
            score = 1.0 - (abs((float(desired_syllables) - float(syllables))) + abs(
                (float(desired_rhyme) - float(rhyme)))) - penalty
        except:
            score = 0
        return score
    dataset = []
    for line in generated_lyrics:
        line_list = [line, syllables(line), rhyme(line, rhyme_list)]
        dataset.append(line_list)

    rap = []
    vector_halves = []
    for vector in vectors:
        vector_halves.append(list(vector[0][0]))
        vector_halves.append(list(vector[0][1]))

    for vector in vector_halves:
        scorelist = []
        for item in dataset:
            line = item[0]

            if len(rap) != 0:
                penalty = last_word_compare(rap, line)
            else:
                penalty = 0
            total_score = calculate_score(vector, item[1], item[2], penalty)
            score_entry = [line, total_score]
            scorelist.append(score_entry)

        fixed_score_list = []
        for score in scorelist:
            fixed_score_list.append(float(score[1]))
        top_score = max(fixed_score_list)
        for item in scorelist:
            if item[1] == top_score:
                rap.append(item[0])
                for i in dataset:
                    if item[0] == i[0]:
                        dataset.remove(i)
                        break
                break
    return rap

def train(x_data, y_data, model):
    model.fit(np.array(x_data).astype(float), np.array(y_data).astype(float), batch_size=2, epochs=10, verbose=1)
    model.save_weights(artist + ".rap")

def main(depth, train_mode):
    model = create_NN(depth, False)
    NN_model = create_NN(depth, True)
    text_file = "Eminem.txt"
    NN_text = "diss_track.txt"
    if train_mode == True:
        bars = split_lyrics_file(text_file)
        NN_bars = split_lyrics_file(NN_text)
    if train_mode == False:
        bars = generate_lyrics(text_file)
        NN_bars = generate_lyrics(NN_text)
    rhyme_list = rhymeindex(bars)
    if train_mode == True:
        x_data, y_data = build_dataset(bars, rhyme_list)
        train(x_data, y_data, model)
        NN_x_data, NN_y_data = build_dataset(NN_bars, rhyme_list)
        train(NN_x_data, NN_y_data, NN_model)
    if train_mode == False:
        vectors = compose_rap(bars, rhyme_list, text_file, model)
        rap = vectors_into_song(vectors, bars, rhyme_list)
        f = open(rap_file, "a")
        for bar in rap:
            bar = str(bar)
            if clean_mode == True: 
                clean = pf.censor(bar)
            else:
                clean = bar
            f.write(clean)
            f.write("\n")
        vectors = compose_rap(NN_bars, rhyme_list, NN_text, model)
        rap = vectors_into_song(vectors, NN_bars, rhyme_list)
        f = open(rap_file, "a")
        for bar in rap:
            bar = str(bar)
            if clean_mode == True: 
                clean = pf.censor(bar)
            else:
                clean = bar
            f.write(clean)
            f.write("\n")

main(depth, train_mode)
