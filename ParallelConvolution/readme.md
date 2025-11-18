# C++ Convolution Benchmark

This project is completely handled by Visual Studio IDE. You must import the `ParallelConvolution.sln` project file provided. You **MUST** keep the `.vcxproj` files as they keep **CRITICAL** compilation settings that will make or break the program.

If you are not using Visual Studio IDE, do **NOT** expect your compiled program to work as intended. Many fail-safe are in place to prevent crashing the libraries, but that also means incorrect compilation settings will result in buggy program.

Your Visual Studio must set up to handle C++ Desktop Development.

## OS Support

- Windows 11: Perfect

- Windows 10: Untested (please update to Windows 11)

- Linux: Untested (Any distribution)

- MacOS: Untested (Any versions)

## Compile and Run

After you have imported the project into Visual Studio, right click the project in `Solution Explorer` and click on `Build` (should be the first option).

You can either run the compiled project by clicking the green start button on top or directly launch `ParallelConvolution.exe` located in `./x64/Debug` folder via Powershell or Command Prompt.

## Alternatives

Look at `results_ox.txt` and `results_ox_avx` in `/data` folder to see my output of the program.

### If you are brave...

1. Install VS Code, Cursor, Copilot, Junie, or any other agentic AI coding IDE.

2. Open the AI agent chat window.

3. Tell it to figure out how to compile this program for your operating system and generate all the necessary command files.

4. Good luck!