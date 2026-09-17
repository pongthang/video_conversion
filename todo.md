I will give you a youtube video link or video which content video in chinese language. I want to generate the caption in English and add a tts in english too. make an automated pipeline to convert a chinese audio video to a video with English speaking and have caption overlay with english transcipt. use opensource model. implement the complete pipeline , consider the pipeline can be run in this laptop. you can run one by one, first extract audio, transcribe it using a opensource model ASR like whisper. then translate it using a translation model , then generate audio in English using a tts. then compile it to make the final video with english audio and subtitle. suggest is there any better way of implementation. if the video is too long,you can cut into multiple chunks of 20 mins then run it. 

make python env here and install the necessary package and download the necesary models here and use it. write proper script so that I can setup in other machine the whole pipeline easily. 

final program can be like this.
this will setup all and ready to run
./setup.sh 

this is to convert.

./convert_video.sh input.mp4 -out output.mp4

then I should get the final outout at output.mp4