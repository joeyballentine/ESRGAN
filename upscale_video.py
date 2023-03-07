import os
import shutil
from upscale import Upscale
import subprocess
import datetime
from pathlib import Path
from pymediainfo import MediaInfo
from math import ceil
from sys import argv


def run_bash(bash_command: str):
    bash_command_list = bash_command.split()
    process = subprocess.Popen(bash_command_list, stdout=subprocess.PIPE)
    output, error = process.communicate()


def format_seconds(seconds):
    td = datetime.timedelta(seconds=seconds)
    return ':'.join(str(td).split(':'))


def get_cut_video(video: str, second_start: int, second_elapsed: int = 5):
    if os.path.isdir('input'):
        shutil.rmtree('input')
    os.makedirs('input')
    if os.path.isdir('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    bash_command = "ffmpeg -loglevel warning -i " + video + ' -ss ' + format_seconds(second_start) \
        +  ' -to ' + format_seconds(second_start + second_elapsed) + ' input/%04d.bmp'
    print(bash_command)
    run_bash(bash_command)
    return second_start + second_elapsed


def get_frames_video(last_frame: int = 5):
    if os.path.isdir('input'):
        shutil.rmtree('input')
    os.makedirs('input')
    if os.path.isdir('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    list_frames = os.listdir('input_temp')
    list_frames.sort()
    last_frame = min(len(list_frames), last_frame)
    frames = list_frames[:last_frame]
    for f in frames:
        shutil.move(os.path.join('input_temp', f),
                    os.path.join('input', f))


def get_video_from_frames(counter, batch_frames):
    bash_command = 'ffmpeg -loglevel warning -r 30 -start_number ' \
        + str(counter * batch_frames) \
        + ' -i output/%04d.bmp -vcodec libx265 -crf 10 -pix_fmt yuv420p output_temp/' \
        + str(counter) +  '.mp4'
    run_bash(bash_command)


def join_videos(folder='test', video='output.mp4'):
    filenames = os.listdir(folder)
    filenames.sort(key=lambda d: int(d.split('.')[0]))
    with open('temp_file.txt', 'w') as w:
        for filename in filenames:
            w.write('file ' + os.path.join(folder, filename) + '\n')
    final_video = video.split('.')[0] + '_ESRGAN.mp4'
    command = 'ffmpeg -loglevel warning -f concat -safe 0 -i temp_file.txt ' \
        + '-c copy -y -vcodec libx264 -crf 10 -pix_fmt yuv420p ' \
        + final_video
    run_bash(command)
    os.remove('temp_file.txt')
    return final_video


def extract_all_frames(video):
    if os.path.isdir('input_temp'):
        shutil.rmtree('input_temp')
    os.makedirs('input_temp')
    bash_command = "ffmpeg -loglevel warning -i " + video + ' input_temp/%04d.bmp'
    run_bash(bash_command)



def transfer_audio(source_video, target_video):
    tempAudioFileName = "temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        run_bash('ffmpeg -y -i ' + source_video + ' -c:a copy -vn ' + tempAudioFileName)

    targetNoAudio = os.path.splitext(target_video)[0] + "_noaudio" + os.path.splitext(target_video)[1]
    shutil.copy(target_video, targetNoAudio)
    # combine audio file and new video file
    run_bash('ffmpeg -y -i {} -i {} -c copy {}'.format(targetNoAudio, tempAudioFileName, target_video))

    if os.path.getsize(target_video) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        run_bash('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(source_video, tempAudioFileName))
        run_bash('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, target_video))
        if (os.path.getsize(target_video) == 0): # if aac is not supported by selected format
            shutil.move(targetNoAudio, target_video)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")


def get_resolution(video):
    media_info = MediaInfo.parse(video)
    video = media_info.tracks[1]
    width = video.width
    height = video.height
    return width, height


if __name__ == '__main__':
    video = argv[1]  #'big_apple_cut.mp4'
    model = 'models/deindeo_x4.pth'
    batch_frames = 30 * 3

    extract_all_frames(video)
    ups = Upscale(model=model,
                  input=Path('input'),
                  output=Path('output'),
                  delete_input=True,
                  fp16=True,
                  cache_max_split_depth=True)

    if os.path.isdir('output_temp'):
        shutil.rmtree('output_temp')
    os.makedirs('output_temp')

    counter = 1
    batches = ceil(len(os.listdir('input_temp')) / batch_frames)
    print('There is ' + str(batches) + ' batches')
    while len(os.listdir('input_temp')) != 0:
        print('Processing batch ' + str(counter) + '...')
        get_frames_video(batch_frames)
        ups.run()
        get_video_from_frames(counter, batch_frames)
        counter += 1

    final_video = join_videos('output_temp', video)
    transfer_audio(video, final_video)
