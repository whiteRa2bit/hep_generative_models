#include <fstream>

void read_detector() {
    ofstream myfile;
    myfile.open("../source_data/z.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> z(reader, "z");

    while (reader.Next()) {
        myfile << *z << endl;
    }
    myfile.close();
}
