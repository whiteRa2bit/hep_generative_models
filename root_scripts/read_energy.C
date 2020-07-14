#include <fstream>
#include <TString.h>

void read_energy() {
    TString DATA_PATH = "../source_data/angle_0_2_GeV/";
    TString write_file = "energy.txt";
    TString read_file = "hybrid0.root";

    TString write_path = DATA_PATH + write_file;
    TString read_path = DATA_PATH + read_file;

    ofstream myfile;
    myfile.open(write_path);

    TFile file(read_path);
    TTreeReader reader("hybrid;128", &file);
    TTreeReaderValue<float> energy(reader, "PhotonEnergy");

    while (reader.Next()) {
        myfile << *energy << endl;
    }
    myfile.close();
}
