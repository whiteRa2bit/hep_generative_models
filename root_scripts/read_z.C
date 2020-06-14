#include <fstream>

void read_z() {
    TString data_folder = "../source_data/angle_0_100_GeV/";
    TString write_file = "z.txt";
    TString read_file = "hybrid0.root";

    TString write_path = data_folder + write_file;
    TString read_path = data_folder + read_file;

    ofstream myfile;
    myfile.open(write_path);

    TFile file(read_path);
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> z(reader, "z");

    while (reader.Next()) {
        myfile << *z << endl;
    }
    myfile.close();
}
