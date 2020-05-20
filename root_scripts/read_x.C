#include <fstream>

void read_x() {
    ofstream myfile;
    myfile.open("../source_data/x.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> x(reader, "x");

    while (reader.Next()) {
        myfile << *x << endl;
    }
    myfile.close();
}
