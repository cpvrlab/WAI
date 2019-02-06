#include <inttypes.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    r32;
typedef double   r64;

typedef int32_t bool32;

#define kilobytes(value) ((value)*1024LL)
#define megabytes(value) (kilobytes(value) * 1024LL)
#define gigabytes(value) (megabytes(value) * 1024LL)
#define terabytes(value) (gigabytes(value) * 1024LL)

#if BUILD_DEBUG
#    define Assert(expression) \
        if (!(expression)) { *(int*)0 = 0; }
#else
#    define Assert(expression)
#endif

#define PI 3.1415926535897932384626433832795
#define DEG2RAD ((r32)(PI / 180.0f))

struct FrameBuffer
{
    void* memory;
    i32   width, height;
    i32   bytesPerPixel;
    i32   pitch;
};

#define PATCH_SIZE 31
#define HALF_PATCH_SIZE 15
#define EDGE_THRESHOLD 19
#define ORB_DESCRIPTOR_COUNT 32

#include "wai_orbpattern.h"

static inline float computeKeypointAngle(FrameBuffer*            buffer,
                                         r32                     x,
                                         r32                     y,
                                         const std::vector<int>& u_max)
{
    int m_01 = 0;
    int m_10 = 0;

    const u8* center = ((u8*)buffer->memory) + cvRound(y) * buffer->pitch + cvRound(x) * buffer->bytesPerPixel;

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    i32 pitch = buffer->pitch;
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d     = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus  = center[u + v * pitch];
            int val_minus = center[u - v * pitch];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}

static inline u8 getOrbPatternValue(const u8*        center,
                                    const i32        bufferPitch,
                                    const r32        a,
                                    const r32        b,
                                    const cv::Point* pattern,
                                    const i32        index)
{
    u8 result;

    result = *(center +
               cvRound(pattern[index].x * b + pattern[index].y * a) * bufferPitch +
               cvRound(pattern[index].x * a - pattern[index].y * b));

    return result;
}

static void computeOrbDescriptor(FrameBuffer*        buffer,
                                 const cv::KeyPoint* keyPoint,
                                 const cv::Point*    pattern,
                                 u8*                 desc)
{
    Assert(buffer->bytesPerPixel == 1);

    float angle = (r32)keyPoint->angle * DEG2RAD;
    float a = (r32)cos(angle), b = (r32)sin(angle);

    const u8* center = ((u8*)buffer->memory) + cvRound(keyPoint->pt.y) * buffer->pitch + cvRound(keyPoint->pt.x) * buffer->bytesPerPixel;
    const i32 pitch  = buffer->pitch;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * pitch + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    for (int i = 0; i < ORB_DESCRIPTOR_COUNT; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0  = getOrbPatternValue(center, pitch, a, b, pattern, 0);
        t1  = getOrbPatternValue(center, pitch, a, b, pattern, 1);
        val = t0 < t1;
        t0  = getOrbPatternValue(center, pitch, a, b, pattern, 2);
        t1  = getOrbPatternValue(center, pitch, a, b, pattern, 3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (u8)val;
    }

#undef GET_VALUE
}

std::vector<cv::KeyPoint> detectFastCorners(FrameBuffer* buffer,
                                            i32          threshold)
{
    Assert(buffer->bytesPerPixel == 1);

    cv::Mat                   cvImg = cv::Mat(buffer->width, buffer->height, CV_8UC1, buffer->memory, buffer->pitch);
    std::vector<cv::KeyPoint> result;

    cv::FAST(cvImg, result, threshold, true);

    return result;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int descriptorDistance(const u8* a, const u8* b)
{
    const i32* pa = (i32*)a;
    const i32* pb = (i32*)b;

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v              = v - ((v >> 1) & 0x55555555);
        v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
