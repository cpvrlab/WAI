:: Add C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE to PATH variable so visual studio is found
set PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE"

set G2O_INSTALL_DIR=%cd%\thirdparty\prebuilt\win64_g2o\bin
set OPENCV_INSTALL_DIR=%cd%\thirdparty\prebuilt\win64_opencv_3.4.1\lib

:: add path variable to find dlls
set PATH=%PATH%;%OPENCV_INSTALL_DIR%\x64\vc15
set PATH=%PATH%;%G2O_INSTALL_DIR%

echo %PATH%

devenv.exe BUILD\WAI.sln

pause