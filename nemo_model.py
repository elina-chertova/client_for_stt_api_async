import nemo
import nemo.collections.asr as nemo_asr
import wget
import csv
import ssl
import sys
import deepspeech
import numpy as np
from pydub import AudioSegment
from flask_socketio import join_room, leave_room
async_mode = None
import os
import urllib.request
import wave
import socketio
import soundfile
from flask import Flask, request, send_file
from flask import jsonify, make_response
import eventlet
import ctc_decoders

eventlet.monkey_patch()


ssl._create_default_https_context = ssl._create_unverified_context

sio = socketio.Server(async_mode=async_mode)
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.config['SECRET_KEY'] = 'secret'


asr_model = nemo_asr.models.EncDecCTCModel.restore_from('QuartzNet15x5_golos.nemo')


HTTP_SERVER_PORT = 5000


@sio.event
def connect(sid, auth):
    print('connect ', sid)


@sio.event
def disconnect(sid):
    print('disconnected', sid)


@sio.on('my error event')
def on_my_event(data):
    raise RuntimeError()


from flask_socketio import join_room, leave_room


@sio.on('join')
def begin_chat(sid, data):
    """
    Функция для присоединения пользователя к комнате.
            Параметры:
                    sid: идентификатор
                    data: данные, которые отправляет клиент
    """
    path = "id_file.csv"
    id_ = [[data['ID']]]
    csv_writer(id_, path)
    print(data['username'] + ' joined the room ' + data['room'])
    sio.enter_room(sid, data['room'])


@sio.on('leave')
def exit_chat(sid, data):
    """
    Функция для выхода пользователя из комнаты.
            Параметры:
                    sid: идентификатор
                    data: данные, которые отправляет клиент
    """
    print(data['username'] + 'left the room' + data['room'])
    sio.leave_room(sid, data['room'])


@sio.on('request_for_text')
def request_for_text(sid, data):
    link = data['req_url'] + '/socket.io/request_for_text/output.txt'
    print(link)
    sio.emit('response_for_text', {'text': link}, broadcast=True)


@sio.on('data_from_client')
def data_from_client(sid, data):
    """
    Основная  функция создания субтитров.

            Параметры:
                    sid:  группа пользователя
                    data:  данные, полученные от клиента
            Отправляет клиенту куски текста с течением транскрибации.

    """
    url = data['url']
    user_id = data['user_ID']  # идентификатор пользователя
    extension = data.get('audio_format')
    errors(data, extension)  # проверка на ошибки

    google_disk_links(url, extension)
    # загрузка файлов с гугл диска
    # urllib.request.urlretrieve(url, "audio_for_stt." + extension)  # download url to a local path
    # if extension != 'wav':
    #   convert_to_wav(extension)

    normal_format()
    # Функция для обработки записи

    w = wave.open('audio_for_stt.wav', 'r')

    frames = w.getnframes()
    buffer = w.readframes(frames)
    os.remove("audio_for_stt.wav")
    ds_stream = model.createStream()
    buffer_len = len(buffer)

    batch_size = 16384
    text = ''
    offset = 0

    beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=list(asr_model.decoder.vocabulary),
        beam_width=16,
        alpha=2, beta=1.5,
        lm_path='kenlms/lm_golos.binary',
        num_cpus=1,
        cutoff_prob=1.0, cutoff_top_n=40,
        input_tensor=True)

    hyps = infer_beam_search_lm(['audio_for_stt.wav'], asr_model, beam_search_lm)
    print(hyps)


    while offset < buffer_len:
        end_offset = offset + batch_size
        chunk = buffer[offset:end_offset]
        data16 = np.frombuffer(chunk, dtype=np.int16)
        ds_stream.feedAudioContent(data16)
        text = ds_stream.intermediateDecode()
        print(text)
        sio.emit('data_from_server', {'text': text}, broadcast=True)
        sio.sleep()
        offset = end_offset

    with open("output.txt", "a") as f:  # запись id пользователя и итогового текста
        f.write(user_id + ': ' + text + '\n')
    f.close()


def normal_format():
    """
    Функция для обработки записи.
    Перевод записи в следующий формат: .wav, 16kNz, 16 bit
    """
    w = wave.open('audio_for_stt.wav', 'r')
    rate = w.getframerate()  # частота дискретизации

    if rate != 16000:
        w.close()
        os.rename("audio_for_stt.wav", "old.wav")
        os.system("sox " + '-v 0.98' + ' old.wav' + " -G -r 16000 -c 1 -b 16 " + 'audio_for_stt.wav')
        os.remove("old.wav")


def errors(data, extension):
    """Функция для обработки ошибок."""
    if not data or not 'url' in data:
        return make_response(jsonify({"error": "There is no link."}), 400)

    if extension not in ['wav', 'opus', 'mp3']:
        return make_response(jsonify({"error": "The format " + extension + " is not acceptable."}), 400)


def convert_to_wav(type_of_file):
    """
    Функция для конвертирования записей формата .mp3 или .opus в  формат .wav

            Параметры:
                    type_of_file:  формат аудиозаписи (.mp3 или .opus)
    """
    if type_of_file == 'opus':
        data, samplerate = soundfile.read('audio_for_stt.opus')
        soundfile.write('audio_for_stt.wav', data, samplerate, subtype='PCM_16')

    if type_of_file == 'mp3':
        sound = AudioSegment.from_mp3('audio_for_stt.mp3')
        sound.set_frame_rate(16000)
        sound.export('audio_for_stt.wav', format="wav")

    os.remove("audio_for_stt." + type_of_file)


def google_disk_links(url, extension):
    """
    Функция для обработки гугловский ссылок и их загрузки локально.

            Параметры:
                    url:  ссылка на запись
                    extension:  формат аудиозаписи (.mp3 или .opus)
    """
    file_id_num = url.find("/d/") + 3
    file_id = url[file_id_num: url.find("/", file_id_num)]
    link_to_download = 'https://drive.google.com/uc?export=download&id=' + file_id
    urllib.request.urlretrieve(link_to_download, "audio_for_stt." + extension)  # download url to a local file
    if extension != 'wav':
        convert_to_wav(extension)


def infer_beam_search_lm(files, asr_model, beam_search_lm):
    hyps = []
    logits = tuple(torch.tensor(asr_model.transcribe(files, batch_size=20, logprobs=True)))
    log_probs_length = torch.tensor([logit.shape[0] for logit in logits])
    logits_tensor = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True)
    for j in range(logits_tensor.shape[0]):
        best_hyp = beam_search_lm.forward(log_probs = logits_tensor[j].unsqueeze(0),
                                          log_probs_length=log_probs_length[j].unsqueeze(0))[0][0][1]
        hyps.append(best_hyp)
    return hyps





def number_of_channels(w):
    """
    Функция получения количества каналов в wav файлах.

            Параметры:
                    w:  название файла
    """
    sound = AudioSegment.from_file(w)
    channel_count = sound.channels
    return channel_count


@app.route('/transcribe', methods=['GET'])
def get_link():
    """
    Фунция для отправки клиенту ссылки на файл, если у того есть доступ.

            Возвращаемое значение:
                    link:  ссылка на файл
    """

    data = request.json
    link = 'У вас нет доступа к данным'
    print(data.get('ID'))
    for line in csv.reader(open('id_file.csv')):
        if data["ID"] in set(line):  # пересечение множеств
            link = request.url + '/output.txt'
            print('link =', link)

    return make_response(jsonify({"link": link}))


@app.route('/transcribe/output.txt', methods=['GET'])  # загрузка файла
def download():
    """Функция для загрузки файла по ссылке"""
    return send_file('output.txt', attachment_filename="output.txt", as_attachment=True)


def csv_writer(data, path):
    """
    Функция для записи идентификатора в csv файл.
            Входные данные:
                    data:  идентификатор пользователя
                    path:  путь к файлу
    """
    with open(path, "a", newline='\n') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


if __name__ == '__main__':
    print(sio.async_mode)
    import eventlet
    import eventlet.wsgi

    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', HTTP_SERVER_PORT)), app)
