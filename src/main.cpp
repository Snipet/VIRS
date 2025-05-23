#include <stdio.h>
#include "args.hxx"

int main(int argc, char *argv[]) {
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
    }
    return 0;
}