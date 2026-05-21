from gtts import gTTS

text = """
Prompt:
I want to learn computer vision and reinforcement learning for self driving cars

Current skills:
Python, Linear Algebra

Student level:
Intermediate

Career goal:
Autonomous Systems Engineer
"""

tts = gTTS(text=text, lang='en')

tts.save("input_audio.mp3")

print("Audio file saved as input_audio.mp3")