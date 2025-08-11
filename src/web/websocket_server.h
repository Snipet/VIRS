#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <memory>
#include <string>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace asio = boost::asio;
namespace fs = boost::filesystem;
using tcp = boost::asio::ip::tcp;


class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
public:
    websocket::stream<tcp::socket> ws_;
    beast::flat_buffer buffer_;

    explicit WebSocketSession(tcp::socket&& socket) : ws_(std::move(socket)) {}

    void run() {
        ws_.async_accept(
            beast::bind_front_handler(&WebSocketSession::onAccept, shared_from_this()));
    }

private:

    void onAccept(beast::error_code ec) {
        if (ec) return fail(ec, "accept");
        doRead();
    }   

    void doRead() {
        ws_.async_read(buffer_,
            beast::bind_front_handler(&WebSocketSession::onRead, shared_from_this()));
    }   
    
    void onRead(beast::error_code ec, std::size_t) {
        if (ec == websocket::error::closed)
            return;

        if (ec) return fail(ec, "read");

        std::string msg = beast::buffers_to_string(buffer_.data());
        std::cout << "Received: " << msg << "\n";

        ws_.text(ws_.got_text());
        ws_.async_write(
            asio::buffer("Echo: " + msg),
            beast::bind_front_handler(&WebSocketSession::onWrite, shared_from_this()));
    }
    
    void onWrite(beast::error_code ec, std::size_t) {
        if (ec) return fail(ec, "write");

        buffer_.consume(buffer_.size());
        doRead();
    }
    
    void fail(beast::error_code ec, char const* what) {
        std::cerr << what << ": " << ec.message() << "\n";
    }    

};

class WebSocketServer {
public:
    WebSocketServer() = default;
    ~WebSocketServer() = default;

bool prepareAcceptor(asio::io_context& ioc,
                     std::shared_ptr<tcp::acceptor>& acc,
                     tcp::endpoint ep)
{
    beast::error_code ec;

    acc = std::make_shared<tcp::acceptor>(ioc);
    acc->open(ep.protocol(), ec);
    if (ec) return (std::cerr << "open: "  << ec.message() << '\n'), false;

    acc->set_option(asio::socket_base::reuse_address(true), ec);
    if (ec) return (std::cerr << "set_option: "  << ec.message() << '\n'), false;

    acc->bind(ep, ec);
    if (ec) return (std::cerr << "bind: "  << ec.message() << '\n'), false;

    acc->listen(asio::socket_base::max_listen_connections, ec);
    if (ec) return (std::cerr << "listen: " << ec.message() << '\n'), false;

    return true;
}

void run() {
    asio::io_context ioc{1};
    tcp::endpoint endpoint(asio::ip::address_v4::any(), 8080);

    std::shared_ptr<tcp::acceptor> acceptor;
    if (!prepareAcceptor(ioc, acceptor, endpoint)) {
        return;
    }

    std::cout << "WebSocket server running on port 8080\n";

    // Accept loop
    std::function<void()> doAccept;
    doAccept = [acceptor, &doAccept]() {
        acceptor->async_accept([acceptor, &doAccept](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                std::make_shared<WebSocketSession>(std::move(socket))->run();
            } else {
                std::cerr << "accept: " << ec.message() << "\n";
            }
            doAccept();  // continue accepting
        });
    };
    doAccept();

    ioc.run();
}
    

private:
// Accepts incoming connections
void doListen(asio::io_context& ioc, tcp::endpoint endpoint) {
    auto acceptor = std::make_shared<tcp::acceptor>(ioc);

    beast::error_code ec;
    acceptor->open(endpoint.protocol(), ec);
    acceptor->set_option(asio::socket_base::reuse_address(true));
    acceptor->bind(endpoint, ec);
    acceptor->listen(asio::socket_base::max_listen_connections, ec);

    std::function<void()> doAccept;
    doAccept = [&]() {
        acceptor->async_accept([&, acceptor](beast::error_code ec, tcp::socket socket) {
            if (!ec)
                std::make_shared<WebSocketSession>(std::move(socket))->run();
            doAccept();
        });
    };
    doAccept();
}
};
