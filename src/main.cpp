#include <stdio.h>
#include "args.hxx"
#include "simulator/simulator.h"
#include "gui/gui.h"

int main(int argc, char *argv[]) {
    std::cout << "Virtual Impulse Response Synthesizer" << std::endl;
    args::ArgumentParser parser("Virtual Impulse Response Synthesizer");
    args::HelpFlag help(parser, "help", "Display help menu", { 'h', "help" });
    args::Flag gui(parser, "gui", "Run GUI", { "gui" });
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    if (gui) {
        std::cout << "Starting GUI" << std::endl;
        std::unique_ptr<VIRSGUI> gui = std::make_unique<VIRSGUI>();
        gui->showAndRun(600, 600);
    }

    ////Initialize the Simulator
    //Simulator* Simulator = new Simulator(600, 600);
    //if (!Simulator) {
    //    std::cerr << "Failed to initialize Simulator" << std::endl;
    //    return 1;
    //}

    //Simulator->loadObj("C:\\Users\\seant\\Documents\\Projects\\school\\VIRS\\assets\\room8.obj");

    //// Print the Simulator information
    //std::cout << Simulator->toString() << std::endl;

    //Simulator->render("C:\\Users\\seant\\Documents\\Projects\\school\\VIRS\\output\\", 100, 7.f);

    //// Clean up the Simulator
    //delete Simulator;
    //Simulator = nullptr;
    std::cout << "Exiting..." << std::endl;
    return 0;
}
