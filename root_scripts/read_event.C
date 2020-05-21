#include <fstream>

void read_event() {
    ofstream myfile;
    myfile.open("../source_data/event.txt");

    TFile file("../source_data/hybrid0.root");
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<int> event(reader, "event");

    while (reader.Next()) {
        myfile << *event << endl;
    }
    myfile.close();
}
