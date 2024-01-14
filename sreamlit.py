
import streamlit as st
import librosa
import numpy as np
import keras
import asyncio
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd

sample_rate = 22050
n_fft = 2048
n_mfcc = 20
hop_length = 512
num_segments = 10
base_duration = 30


def preprocess(file_path):
    audio_mfcc = []
    audio, sr = librosa.load(file_path, sr=sample_rate)
    full_duration = librosa.get_duration(y=audio, sr=sr)
    full_segment = int(full_duration / base_duration)
    full_sample = full_duration * sr
    full_num_segments = int(full_sample / full_segment)

    sample = base_duration * sr
    num_samples_per_segment = int(sample/num_segments)

    for i in range(full_segment):
        start = full_num_segments * i
        finish = start + full_num_segments
        song = audio[start:finish]
        segment = []
        for j in range(num_segments):
            start_sample = j * num_samples_per_segment
            finish_sample = start_sample + num_samples_per_segment
            segment_audio = song[start_sample:finish_sample]
            mfcc = librosa.feature.mfcc(
                y=segment_audio, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            concanated = np.concatenate(
                (mfcc, delta_mfcc, delta2_mfcc), axis=0)
            concanated = concanated.T
            segment.append(concanated.tolist())
        audio_mfcc.append(segment)

    audio_mfcc = np.array(audio_mfcc)
    return audio_mfcc


def predictions(file_path, model, model_name):
    gtzan_label = ['classical', 'hiphop', 'disco', 'jazz', 'metal',
                   'reggae', 'rock', 'country', 'blues', 'pop']
    ismir_label = ['classical', 'electronic', 'jazz',
                   'metal', 'pop', 'punk', 'rock', 'world']
    # Preprocess audio file
    all_genre_counts = []
    audio_mfcc = preprocess(file_path)
    for i in range(len(audio_mfcc)):
        predictions = model.predict(audio_mfcc[i])
        predicted_genre = np.argmax(predictions, axis=1)
        visulaize_genre = predicted_genre.tolist()
        if model_name == 'gtzan':
            genre_labels = gtzan_label
        else:
            genre_labels = ismir_label
        predicted_genre = np.bincount(predicted_genre).argmax()
        genre_counts = []
        for genre in genre_labels:
            genre_counts.append(visulaize_genre.count(
                genre_labels.index(genre)))
        all_genre_counts.append(genre_counts)
        print(genre_counts)
        st.write("### Segment {}".format(i+1))
        st.write("Predicted Genre: ", genre_labels[predicted_genre])
        st.altair_chart(alt.Chart(pd.DataFrame({'Genre': genre_labels, 'Count': genre_counts})).mark_bar().encode(
            x='Genre', y='Count', color='Genre').properties(width=700, height=400))
        st.markdown('---')

    return all_genre_counts, genre_labels


def overall_predictions(count, label):
    all_genre_counts = count
    genre_labels = label
    overall_genre_counts = []
    for i in range(len(genre_labels)):
        overall_genre_counts.append(sum(
            [item[i] for item in all_genre_counts]))
    predicted_genre = np.argmax(overall_genre_counts)
    st.write("## Final Prediction")
    st.write("With the highest number of votes, the genre is: ",
             genre_labels[predicted_genre])
    st.altair_chart(alt.Chart(pd.DataFrame({'Genre': genre_labels, 'Count': overall_genre_counts})).mark_bar().encode(
                    x='Genre', y='Count', color='Genre').properties(width=700, height=400))


def file_uploader():
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    if uploaded_file is not None:
        st.audio(uploaded_file)
        return uploaded_file
    else:
        return None


def model_selector():
    model_names = ['gtzan', 'ismir']
    model_name = st.selectbox("Select a model", model_names)
    model_path = "models/" + model_name + ".h5"
    return model_path, model_name


async def load_model_async(model_path):
    model = await asyncio.get_event_loop().run_in_executor(None, keras.models.load_model, model_path)
    return model


def plot_genre(genre_counts, genre_labels, segment):
    plt.figure(figsize=(12, 6))
    plt.bar(genre_labels, genre_counts)
    if segment == 0:
        plt.title("Genre pada keseluruhan lagu")
    plt.title("Genre pada sefment ke-{}".format(segment))
    plt.ylabel("Jumlah Potongan Lagu (3 detik)")
    plt.xlabel("Jumlah Data untuk Setiap Genre")
    plt.xticks(rotation=45)  # Untuk memutar label sumbu x agar lebih terbaca
    st.pyplot(plt)


def main():
    st.title("Audio Classification")

    # Display file uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file", type=["wav", "mp3"])

    # Check if file is uploaded
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)

        # Display model selector
        model_path, model_name = model_selector()

        # Check if a model is selected
        if model_path:
            # Use a button to trigger model loading asynchronously
            if st.button("Load Model"):
                st.text("Loading model... Please wait.")
                # Use asyncio.gather to run tasks concurrently
                model = asyncio.run(load_model_async(model_path))
                st.text("Predicting...")
                st.markdown('---')
                result = predictions(uploaded_file, model, model_name)

                # Display overall genre prediction
                overall_predictions(result[0], result[1])
                print(result[1])


if __name__ == '__main__':
    main()
