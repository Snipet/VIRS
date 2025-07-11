#include <stdio.h>
#include "args.hxx"
#include "simulator/simulator.h"
#include "simulator/1D_simulation.h"
#include "util/logger.h"

int main(int argc, char *argv[]) {
    args::ArgumentParser parser("Virtual Impulse Response Synthesizer");
    args::Group commands(parser, "commands");
    args::Command simulateCommand(commands, "simulate", "Run a simulation");
    args::Command renderSceneCommand(commands, "renderscene", "Render a scene to an image file");
    args::Command simulate1DCommand(commands, "simulate1D", "Run a 1D simulation)");
    args::Group arguments(parser, "arguments",args::Group::Validators::DontCare, args::Options::Global);
    args::HelpFlag help(arguments, "help", "Display help menu", { 'h', "help" });
    args::ValueFlag<std::string> outputFile(arguments, "output", "Output file name", { 'o', "output" });
    args::ValueFlag<std::string> inputFile(arguments, "input", "Input file name", { 'i', "input" });
    
    Logger::getInstance().init(
        static_cast<unsigned int>(
            LOGGER_ERROR | LOGGER_WARNING | LOGGER_DEBUG | LOGGER_NONCRITIAL_INFO | LOGGER_CRITICAL_INFO | LOGGER_SUCCESS
        )
    );

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
    if( renderSceneCommand ) {
        Simulator* simulator = new Simulator(1920, 1080);
        if (inputFile) {
            simulator->loadConfig(args::get(inputFile), false);
        } else {
            Logger::getInstance().log("No input file specified for rendering.", LOGGER_ERROR);
            delete simulator;
            return 1;
        }

        if( outputFile ) {
            std::cout << "Rendering scene to file: " << args::get(outputFile) << std::endl;
            simulator->renderImageToFile({7, 7, 7}, args::get(outputFile));
        } else {
            Logger::getInstance().log("No output file specified for rendering.", LOGGER_ERROR);
            delete simulator;
            return 1;
        }
        delete simulator;
    }

    //RUN SIMULATION
    if( simulateCommand ) {
        Simulator* simulator = new Simulator(800, 600);
        if(inputFile){
            if(simulator->loadConfig(args::get(inputFile))){
                std::cout << "Loaded simulation configuration from: " << args::get(inputFile) << std::endl;
            }else{
                Logger::getInstance().log("Failed to load simulation configuration from: " + args::get(inputFile), LOGGER_ERROR);
                delete simulator;
                return 1;
            }
        }else{
            Logger::getInstance().log("No input file specified for simulation.", LOGGER_ERROR);
            return 1;
        }

        simulator->simulate();
        delete simulator;
    }

    if(simulate1DCommand){
        Simulation1D* simulation1D = new Simulation1D();
        simulation1D->simulate(20000);
        delete simulation1D;
    }
    return 0;
}