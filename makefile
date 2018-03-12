buat:
	gcc `pkg-config --cflags gtk+-3.0` -o uas GUI_UAS.c `pkg-config --libs gtk+-3.0` genann.c
