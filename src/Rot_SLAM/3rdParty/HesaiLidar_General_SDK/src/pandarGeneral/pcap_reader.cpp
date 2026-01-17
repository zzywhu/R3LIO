#include "pandarGeneral/pcap_reader.h"

#define PKT_HEADER_SIZE (42)

PcapReader::PcapReader(std::string path, std::string frame_id) {
  pcapPath   = path;
  loop       = false;
  finished   = false;
  parse_thr_ = NULL;
}

PcapReader::~PcapReader() {
  stop();
}

void PcapReader::start(std::function<void(const uint8_t*, const int)> callback) {
  stop();

  this->callback = callback;
  loop           = true;
  finished       = false;

  parse_thr_ = new std::thread(std::bind(&PcapReader::parsePcap, this));
}

void PcapReader::stop() {
  loop = false;
  finished = true;

  if (parse_thr_) {
    // parse_thr_->interrupt();  // no interrupt before c++20
    parse_thr_->join();
    delete parse_thr_;
    parse_thr_ = NULL;
  }
}

bool PcapReader::isFinished() {
  return finished;
}

void PcapReader::parsePcap() {
  pcap_t *pcapFile = NULL;
  char pcapBuf[PCAP_ERRBUF_SIZE];
  struct bpf_program filter;
  pcap_pkthdr *pktHeader;
  const unsigned char *packetBuf;

  pcapFile = pcap_open_offline(pcapPath.c_str(), pcapBuf);

  if (NULL == pcapFile) {
    printf("open pcap file %s failed\n", pcapPath.c_str());
    return;
  }

  if (pcap_compile(pcapFile, &filter, "udp", 0, 0xffffffff) == -1) {
    printf("compile pcap file failed\n");
    return;
  }

  if (pcap_setfilter(pcapFile, &filter) == -1) {
    printf("pcap set filter failed\n");
    return;
  }

  if (NULL == callback) {
    printf("pcap read callback is null\n");
    return;
  }
  while (pcap_next_ex(pcapFile, &pktHeader, &packetBuf) >= 0 && loop) {
    finished = false;
    // std::this_thread::interruption_point();  // no interrupt before c++20
    const uint8_t *packet = packetBuf + PKT_HEADER_SIZE;
    int pktSize = pktHeader->len - PKT_HEADER_SIZE;
    callback(packet, pktSize);
  }

  if (pcapFile != NULL) {
    pcap_close(pcapFile);
    finished = true;
    pcapFile = NULL;
  }
}
