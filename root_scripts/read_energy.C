#include <fstream>

void read_energy() {
    ofstream myfile;
    myfile.open("../source_data/energy.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> energy(reader, "PhotonEnergy");

    while (reader.Next()) {
        myfile << *energy << endl;
    }
    myfile.close();
}
