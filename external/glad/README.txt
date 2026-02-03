This directory is expected to contain GLAD sources:

  include/glad/glad.h
  include/KHR/khrplatform.h
  src/glad.c

Generate them from https://glad.dav1d.de/ with:
  - API: OpenGL
  - Profile: Core
  - Version: 4.6
  - Language: C/C++
  - Generate a loader: checked

Then place the generated 'include' and 'src' folders here.
