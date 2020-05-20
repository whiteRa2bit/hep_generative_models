#include <fstream>

void read_y() {
    ofstream myfile;
    myfile.open("../source_data/y.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> y(reader, "y");

    while (reader.Next()) {
        myfile << *y << endl;
    }
    myfile.close();
}
