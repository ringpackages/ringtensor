cls
setlocal enableextensions enabledelayedexpansion
call ../../language/build/locatevc.bat
cl /c %ringcflags% ring_tensor.c -I"..\..\language\include"
link %ringldflags% ring_tensor.obj  ..\..\lib\ring.lib /DLL /OUT:..\..\bin\ring_tensor.dll /SUBSYSTEM:CONSOLE,"5.01" 
del ring_tensor.obj
endlocal