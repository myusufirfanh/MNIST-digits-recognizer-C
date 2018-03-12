buat:
	gcc `pkg-config --cflags gtk+-3.0` -o mnist_nn GUI.c `pkg-config --libs gtk+-3.0` genann.c
