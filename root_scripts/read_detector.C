#include <fstream>

void read_detector() {
    ofstream myfile;
    myfile.open("../source_data/detector.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<int> detector(reader, "detector");

    while (reader.Next()) {
        myfile << *detector << endl;
    }
    myfile.close();
}
