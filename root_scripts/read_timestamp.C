#include <fstream>

void read_timestamp() {
    ofstream myfile;
    myfile.open("../source_data/timestamp.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> timestamp(reader, "timestamp");

    while (reader.Next()) {
        myfile << *timestamp << endl;
    }
    myfile.close();
}
