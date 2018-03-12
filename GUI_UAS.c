#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gtk/gtk.h>

#define MNIST_TRAINING_SET_IMAGE_FILE_NAME "data/train-images-idx3-ubyte" ///< MNIST image training file in the data folder
#define MNIST_TRAINING_SET_LABEL_FILE_NAME "data/train-labels-idx1-ubyte" ///< MNIST label training file in the data folder

#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  ///< MNIST label testing file in the data folder

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "genann.h"

typedef struct{
	GSList *windows;
}App;

GtkWidget *window;
GtkWidget *fixed;
GtkWidget *dialog;
GtkWidget *inputFileEntry;
int width = 28;
int height = 28;
int bpp = 1;
long int magicn = 2051;
long int noimage = 1;
int i,j,k;

int loops = 1;
int samples = 60000;

typedef struct MNIST_ImageFilelHeader {
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
} MNIST_ImageFileHeader;

typedef struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxLabels;
} MNIST_LabelFileHeader;

uint32_t flipBytes(uint32_t n){
    
    uint32_t b0,b1,b2,b3;
    
    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;
    
    return (b0 | b1 | b2 | b3);
    
}

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){
    
    lfh->magicNumber =0;
    lfh->maxLabels   =0;
    
    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);
    
    fread(&lfh->maxLabels, 4, 1, imageFile);
    lfh->maxLabels = flipBytes(lfh->maxLabels);
    
}

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){
    
    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;
    
    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);
    
    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);
    
    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);
    
    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

FILE *openMNISTImageFile(char *fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);
    
    return imageFile;
}

FILE *openMNISTLabelFile(char *fileName){
    
    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);
    
    return labelFile;
}

void on_window_destroy (GtkWidget *widget, App *app)
{
    app->windows = g_slist_remove (app->windows, widget);
        
	if (g_slist_length (app->windows) == 0){
        g_debug ("Exiting...");
		g_slist_free (app->windows);
		gtk_main_quit ();
	}
}

void training(GtkWidget *widget, gpointer window){
	dialog = gtk_message_dialog_new(NULL,
	        GTK_DIALOG_MODAL,
	        GTK_MESSAGE_INFO,
	        GTK_BUTTONS_OK,	
	        "Training");
	
	FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);

    uint8_t *imageptr;
    uint8_t *classptr;

    imageptr = malloc(sizeof(uint8_t) * samples * width * height);
    classptr = malloc(sizeof(uint8_t) * samples);

    size_t result = fread(imageptr, sizeof(uint8_t), width*height*samples, imageFile);
  
    result = fread(classptr, sizeof(uint8_t), samples, labelFile);

    int jumlahOutputLayer = 10;
    double *input = malloc(sizeof(double) * samples * width * height);
    double *class = malloc(sizeof(double) * samples * jumlahOutputLayer);

    k=0;
    for(j=0;j<samples;j++){
        for(i=0;i<jumlahOutputLayer;i++) {
        class[k] = ((double) classptr[j] == (double) i) ? 1 : 0;
        k++;
        }
    }

    for(i=0; i<width*height*samples; i++) {
        input[i] = ((double) imageptr[i] == (double) 0) ? 0 : 1;
    }

    //Init neural network
    genann *ann = genann_init(width*height, 1, 15, 10);
	
	char count[10];
	printf("Training for %d loops over data.\n", loops);
    for(i = 0; i < loops; i++) {
        for(j = 0; j < samples; j++) {
            genann_train(ann, input + j*784, class + j*10, 0.05);
        }
    }
    itoa(j, count, 10);
    
    printf("\nowatta!\n");
    FILE *NNfile = fopen("NNfile.txt", "w");
    genann_write(ann, NNfile);

    free(imageptr);
    int correct = 0;
    int *maxindex = malloc(sizeof(int) * samples);
    double max = 0;
    for (j = 0; j < samples; j++) {

        const double *guess = genann_run(ann, input + j*784);
        max = 0;
        for(i = 0; i < 10; i++){
            if(guess[i] > max) {
                max = guess[i];
                maxindex[j] = i;
            }
        
        }
        if(classptr[j] == maxindex[j]) {
            correct++;
        }
    }
	
	float r = (correct/ (float) samples) * 100;
	
	gtk_message_dialog_format_secondary_text(
		GTK_MESSAGE_DIALOG(dialog),
		"Jumlah sample adalah %d\nRecognition rate-nya adalah %0.2f", j, r);
	gtk_window_set_position(GTK_WINDOW(dialog), GTK_WIN_POS_CENTER);
	gtk_dialog_run(GTK_DIALOG(dialog));
	gtk_widget_destroy(dialog);	
    
	gtk_widget_show_all (window); 
}

void loadImg(GtkWidget *widget, gpointer window){
	GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
	gint res;
	
	dialog = gtk_file_chooser_dialog_new ("Open File",
	                                      NULL,
	                                      action,
	                                      ("_Cancel"),
	                                      GTK_RESPONSE_CANCEL,
	                                      ("_Open"),
	                                      GTK_RESPONSE_ACCEPT,
	                                      NULL);
	res = gtk_dialog_run (GTK_DIALOG (dialog));
	char *filename;
	if (res == GTK_RESPONSE_ACCEPT){
	    GtkFileChooser *chooser = GTK_FILE_CHOOSER (dialog);
	    filename = gtk_file_chooser_get_filename (chooser);
	}
	uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 0);
	g_free (filename);
	gtk_widget_destroy (dialog);
	
	//Konversi image ke IDX format
    FILE *fp;
    fp = fopen( "imageinIDXformat" , "w" );
    fwrite(&magicn, sizeof(long int), 1, fp);
    fwrite(&noimage, sizeof(long int), 1, fp);
    fwrite(&height, sizeof(long int), 1, fp);
    fwrite(&width, sizeof(long int), 1, fp);
    fwrite(rgb_image, 1, width*height, fp);
    fclose(fp);

    stbi_image_free(rgb_image);
	
    fclose(fp);
    FILE *imageTestFile;
    imageTestFile = openMNISTImageFile("imageinIDXformat");
    uint8_t *imageTestPtr;

    imageTestPtr = malloc(sizeof(uint8_t) * width * height);  
 
    size_t result = fread(imageTestPtr, sizeof(uint8_t), width * height, imageTestFile);
    double *inputImageTest = malloc(sizeof(double) * width * height);

    for(i=0; i< width*height; i++) {
        inputImageTest[i] = ((double) imageTestPtr[i] == (double) 0) ? 0 : 1;
    }
	genann *ann = genann_init(width*height, 1, 15, 10);
	FILE *f = fopen("NNfile.txt", "r");
	ann = genann_read(f);
    const double *guess = genann_run(ann, inputImageTest);
    g_print("\nTesting from image file\n");
    double max = 0;
    int maxindexTest;
    for(i = 0; i < 10; i++){
        if(guess[i] > max) {
            max = guess[i];
            maxindexTest = i;
        }
    }
    char *maxin[5];
    itoa(maxindexTest, maxin, 10);
	dialog = gtk_message_dialog_new(NULL,
	        GTK_DIALOG_MODAL,
	        GTK_MESSAGE_INFO,
	        GTK_BUTTONS_OK,	
	        "Result");
	gtk_message_dialog_format_secondary_text(
		GTK_MESSAGE_DIALOG(dialog),
		"Angka pada gambar adalah %s", maxin);
	gtk_window_set_position(GTK_WINDOW(dialog), GTK_WIN_POS_CENTER);
	gtk_dialog_run(GTK_DIALOG(dialog));
	gtk_widget_destroy(dialog);	
}

void testing(GtkWidget *widget, App *app){
	GtkWidget *backBtn, *judul, *loadFileBtn, *loadImgBtn;
	window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
	app->windows = g_slist_prepend (app->windows, window);
	gtk_window_set_title(GTK_WINDOW(window), "Neural Network Digit Recognition");
	gtk_window_set_default_size(GTK_WINDOW(window), 200, 80);
	gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
	fixed = gtk_fixed_new();
	gtk_container_add(GTK_CONTAINER(window), fixed);
	
	judul = gtk_label_new(NULL);
	gtk_fixed_put(GTK_FIXED(fixed), judul, 79, 10);
	gtk_label_set_markup(GTK_LABEL(judul), "<b>Testing</b>");
	
	loadImgBtn = gtk_button_new_with_label("Load Image");
	gtk_fixed_put(GTK_FIXED(fixed), loadImgBtn, 40, 30);
	gtk_widget_set_size_request(loadImgBtn, 120, 30);
	g_signal_connect(G_OBJECT(loadImgBtn), "clicked",
		G_CALLBACK(loadImg), (gpointer)window);
		
	g_signal_connect(window, "destroy",
		G_CALLBACK(on_window_destroy), app); 
		
	gtk_widget_show_all (window); 
}

void showHelp(GtkWidget *widget, gpointer window){
	dialog = gtk_message_dialog_new(NULL,
	        GTK_DIALOG_DESTROY_WITH_PARENT,
	        GTK_MESSAGE_INFO,
	        GTK_BUTTONS_OK,	
	        "Tekan tombol 'Training' untuk melakukan pelatihan. Tekan tombol 'Testing' untuk melakukan pengujian. Tekan tombol 'Help' untuk menampilkan bantuan. Tekan tombol 'Quit' untuk keluar dari program.");
		gtk_window_set_title(GTK_WINDOW(dialog), "Help");
		gtk_window_set_position(GTK_WINDOW(dialog), GTK_WIN_POS_CENTER);
		gtk_dialog_run(GTK_DIALOG(dialog));
		gtk_widget_destroy(dialog);	
}

int main(int argc, char *argv[]){
	App *app;
	GtkWidget *judul;
	GtkWidget *trainBtn;
	GtkWidget *testBtn;
	GtkWidget *helpBtn;
	GtkWidget *quitBtn;
	
	gtk_init(&argc, &argv);
	
	app = g_slice_new(App);
	window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	app->windows = g_slist_prepend (app->windows, window);
	gtk_window_set_title(GTK_WINDOW(window), "Neural Network Digit Recognition");
	gtk_window_set_default_size(GTK_WINDOW(window), 500, 170);
	gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
	fixed = gtk_fixed_new();
	gtk_container_add(GTK_CONTAINER(window), fixed);
	
	gchar *str = "<b>Neural Network Digit Recognition</b>";
	judul = gtk_label_new(NULL);
	gtk_fixed_put(GTK_FIXED(fixed), judul, 144, 10);
	gtk_label_set_markup(GTK_LABEL(judul), str);
	
	trainBtn = gtk_button_new_with_label("Training");
	gtk_fixed_put(GTK_FIXED(fixed), trainBtn, 120, 50);
  	gtk_widget_set_size_request(trainBtn, 120, 30);
  	
	testBtn = gtk_button_new_with_label("Testing");
	gtk_fixed_put(GTK_FIXED(fixed), testBtn, 250, 50);
  	gtk_widget_set_size_request(testBtn, 120, 30);  	
  	
  	helpBtn =  gtk_button_new_with_label("Help");
  	gtk_fixed_put(GTK_FIXED(fixed), helpBtn, 120, 100);
  	gtk_widget_set_size_request(helpBtn, 120, 30);
	
	quitBtn =  gtk_button_new_with_label("Quit");
  	gtk_fixed_put(GTK_FIXED(fixed), quitBtn, 250, 100);
  	gtk_widget_set_size_request(quitBtn, 120, 30);
	
	g_signal_connect(trainBtn, "clicked",
		G_CALLBACK(training), (gpointer)window);
		
	g_signal_connect(testBtn, "clicked",
		G_CALLBACK(testing), &app);
		
	g_signal_connect(helpBtn, "clicked", 
        G_CALLBACK(showHelp), (gpointer) window); 
        
	g_signal_connect(quitBtn, "clicked",
		G_CALLBACK(gtk_main_quit), NULL);
			
	g_signal_connect(window, "destroy",
		G_CALLBACK(gtk_main_quit), NULL);  
		
	gtk_widget_show_all(window);
  
	gtk_main();
	g_slice_free (App, app);

	return 0;
}
