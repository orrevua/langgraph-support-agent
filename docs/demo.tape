# Record the README GIF with vhs (https://github.com/charmbracelet/vhs):
#   brew install vhs   # or: go install github.com/charmbracelet/vhs@latest
#   vhs docs/demo.tape   -> writes docs/demo.gif
Output docs/demo.gif
Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Type "make demo"
Sleep 500ms
Enter
Sleep 6s

Type "make eval"
Sleep 500ms
Enter
Sleep 6s
