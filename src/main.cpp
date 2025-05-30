#include <stdio.h>
#include "args.hxx"
#include "simulator/simulator.h"

int main(int argc, char *argv[]) {
    args::ArgumentParser parser("Virtual Impulse Response Synthesizer");
    args::HelpFlag help(parser, "help", "Display help menu", { 'h', "help" });
    args::Flag renderScene(parser, "renderscene", "Render a scene to an image file", { "render-scene" });
    args::Flag runSimulation(parser, "runsimulation", "Run a simulation", { "run-simulation" });
    args::ValueFlag<std::string> outputFile(parser, "output", "Output file name", { 'o', "output" });
    args::ValueFlag<std::string> inputFile(parser, "input", "Input file name", { 'i', "input" });
    args::ValueFlag<int> frames(parser, "frames", "Number of simulation frames", { 'f', "frames" }, 100);
    args::ValueFlag<int> outputLayer(parser, "layer", "Output layer for rendering", { 'l', "layer" }, 150);
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

    //RENDER SCENE
    if( renderScene ) {
        Simulator* simulator = new Simulator(800, 600);
        if (inputFile) {
            std::cout << "Loading object from: " << args::get(inputFile) << std::endl;
            simulator->loadObj(args::get(inputFile));
        } else {
            std::cerr << "No input file specified for rendering." << std::endl;
            return 1;
        }

        if( outputFile ) {
            std::cout << "Rendering scene to file: " << args::get(outputFile) << std::endl;
            simulator->renderImageToFile({7, 7, 7}, args::get(outputFile));
        } else {
            std::cerr << "No output file specified for rendering." << std::endl;
            return 1;
        }
        delete simulator;
    }

    //RUN SIMULATION
    if( runSimulation ) {
        Simulator* simulator = new Simulator(800, 600);
        if (inputFile) {
            std::cout << "Loading object from: " << args::get(inputFile) << std::endl;
            simulator->loadObj(args::get(inputFile));
        } else {
            std::cerr << "No input file specified for simulation." << std::endl;
            return 1;
        }

        if(outputLayer) {
            std::cout << "Setting output layer to: " << args::get(outputLayer) << std::endl;
            simulator->setOutputLayer(args::get(outputLayer));
        }

        std::cout << "Running simulation for " << args::get(frames) << " frames." << std::endl;
        simulator->simulate(args::get(frames));
        delete simulator;
    }
    return 0;
}