#include <stdio.h>
#include "args.hxx"
#include "simulator/simulator.h"
#include "simulator/1D_simulation.h"
#include "util/logger.h"
#include "web/websocket_server.h"
#include "AudioFile.h"
#include "util/convolution.h"
#include "util/downsampling_filter_fir.h"
#include "util/downsampling_filter_iir.h"


int main(int argc, char *argv[]) {
    args::ArgumentParser parser("Virtual Impulse Response Synthesizer");
    args::Group commands(parser, "commands");
    args::Command simulateCommand(commands, "simulate", "Run a simulation");
    args::Command renderSceneCommand(commands, "renderscene", "Render a scene to an image file");
    args::Command simulate1DCommand(commands, "simulate1D", "Run a 1D simulation)");
    args::Command webserverCommand(commands, "webserver", "Start the web server");
    args::Command testFilter(commands, "testfilter", "Tests the filter algorithm");
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
        simulation1D->simulate(40000);
        delete simulation1D;
    }

    if(webserverCommand){
        WebSocketServer server;
        server.run();
    }

    if(testFilter){
        std::string path = args::get(inputFile);
        if(path.empty()){
            Logger::getInstance().log("No input file specified for filter testing.", LOGGER_ERROR);
            return 1;
        }
        
        //Open .wav file
        AudioFile<float> wavFile;
        wavFile.load(path);
        size_t numSamples = wavFile.getNumSamplesPerChannel();

        std::cout << "Number of samples: " << numSamples << std::endl;
        std::cout << "Number of channels: " << wavFile.getNumChannels() << std::endl;
        float* output = new float[numSamples];
        std::cout << "Created buffer" << std::endl;
        for(size_t i = 0; i < numSamples; ++i) {
            output[i] = wavFile.samples[0][i]; //Take first channel only
        }

        std::cout << "Loaded samples into buffer" << std::endl;

        //Create filter
        // auto kernel = MinPhaseLPF::design(wavFile.getSampleRate(), 3000.0f, 500.f, 100.f);
        // std::cout << "Designed filter with " << kernel.size() << " taps." << std::endl;

        // convolute(output, output, kernel.data(), numSamples, kernel.size());
        // std::cout << "Applied filter to audio buffer." << std::endl;

        // IIR Filter
        auto sos = designChebyshevI_LP(wavFile.getSampleRate(), 15000.f, 16000.f, 0.5f, 90.0f);
        IIRState state;
        initState(state, sos, 1);
        processBlock(sos, state, output, output, numSamples);

        AudioFile<float> outFile;
        outFile.setSampleRate(wavFile.getSampleRate());
        outFile.setAudioBuffer({std::vector<float>(output, output + numSamples)});
        outFile.save("filtered.wav");
        std::cout << "Saved filtered.wav" << std::endl;
        delete[] output;
    }

    return 0;
}