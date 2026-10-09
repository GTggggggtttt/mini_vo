#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>

#include "mini_vo/dataset.h"


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <euroc_dataset_path>"
            << std::endl;

        return 1;
    }


    const std::string dataset_path =
        argv[1];


    mini_vo::Dataset dataset(
        dataset_path
    );


    if (!dataset.Init())
    {
        std::cerr
            << "Dataset initialization failed."
            << std::endl;

        return 1;
    }


    while (true)
    {
        auto frame =
            dataset.NextFrame();


        if (!frame)
        {
            // 【修正 2】读取失败返回非零退出码，不再误报 Dataset finished。
            if (!dataset.IsFinished())
            {
                std::cerr
                    << "Dataset reading failed. See the error above."
                    << std::endl;
                return 1;
            }

            std::cout
                << "Dataset finished."
                << std::endl;

            break;
        }


        std::cout
            << "Frame ID: "
            << frame->id()
            << "    timestamp: "
            << frame->timestamp()
            << '\n';


        cv::imshow(
            "EuRoC cam0",
            frame->leftImage()
        );


        cv::imshow(
            "EuRoC cam1",
            frame->rightImage()
        );


        const int key =
            cv::waitKey(20);


        if (key == 27)
        {
            break;
        }
    }


    return 0;
}
