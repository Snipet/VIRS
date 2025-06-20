#include <iostream>
#include <string>
#include <mutex>

#define LOGGER_NONCRITIAL_INFO 0x1
#define LOGGER_CRITICAL_INFO 0x2
#define LOGGER_WARNING 0x4
#define LOGGER_ERROR 0x8
#define LOGGER_DEBUG 0x10
#define LOGGER_SUCCESS 0x20


inline void VIRS_PrintRedText(const std::string& text) {
    std::cout << "\033[31m" << text << "\033[0m" << std::endl;
}
inline void VIRS_PrintGreenText(const std::string& text) {
    std::cout << "\033[32m" << text << "\033[0m" << std::endl;
}
inline void VIRS_PrintYellowText(const std::string& text) {
    std::cout << "\033[33m" << text << "\033[0m" << std::endl;
}
inline void VIRS_PrintBlueText(const std::string& text) {
    std::cout << "\033[34m" << text << "\033[0m" << std::endl;
}
inline void VIRS_PrintMagentaText(const std::string& text) {
    std::cout << "\033[35m" << text << "\033[0m" << std::endl;
}
inline void VIRS_PrintCyanText(const std::string& text) {
    std::cout << "\033[36m" << text << "\033[0m" << std::endl;
}

class Logger {
public:


    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    void init(unsigned int verbose_flags) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!initilized) {
            log_verbose_flags = verbose_flags;
            initilized = true;
        }
    }

    void log(std::string message, unsigned int flag){
        std::lock_guard<std::mutex> lock(log_mutex);
        if (initilized && (log_verbose_flags & static_cast<unsigned int>(flag))) {
            if(flag == LOGGER_ERROR) {
                VIRS_PrintRedText(message);
            } else if(flag == LOGGER_WARNING) {
                VIRS_PrintYellowText(message);
            } else if(flag == LOGGER_SUCCESS) {
                VIRS_PrintGreenText(message);
            } else {
                std::cout << message << std::endl;
            }
        }
    }

    


private:
    Logger() = default;
    ~Logger() = default;

    bool initilized = false;
    unsigned int log_verbose_flags;
    std::mutex init_mutex;
    std::mutex log_mutex;

};