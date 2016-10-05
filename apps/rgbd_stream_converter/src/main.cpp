#include <core/common/BasicTypes.h>
#include <core/imageproc/ColorMap.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <core/io/PNGIO.h>
#include <core/vecmath/Range1i.h>
#include <camera_wrappers/RGBDStream.h>
#include <third_party/pystring.h>

using namespace libcgt::camera_wrappers;
using libcgt::core::imageproc::linearRemapToLuminance;

// --> libcgt::core.
#include <iomanip>
#include <sstream>

// T must be an integral type.
template< typename T >
std::string toZeroFilledString( T x, int width )
{
    std::stringstream stream;
    stream << std::setw( width ) << std::setfill( '0' ) << std::internal << x;
    return stream.str();
}

int main( int argc, char* argv[] )
{
    if( argc < 3 )
    {
        printf( "Usage: %s <src.rgbd> <dst.rgbd>\n", argv[ 0 ] );
        return 1;
    }

    RGBDInputStream inputStream( argv[ 1 ], true );
    if( !inputStream.isValid() )
    {
        fprintf( stderr, "Error reading input %s.\n", argv[ 1 ] );
        return 2;
    }

    RGBDOutputStream outputStream( inputStream.metadata(), argv[ 2 ] );
    if( !outputStream.isValid() )
    {
        fprintf( stderr, "Error writing output %s.\n", argv[ 2 ] );
        return 2;
    }

    uint32_t streamId;
    int frameIndex;
    int64_t timestamp;
    Array1DView< const uint8_t > data = inputStream.read( streamId, frameIndex, timestamp );
    while( data.notNull() )
    {
        outputStream.write( streamId, frameIndex, timestamp, data );
        data =
            inputStream.read( streamId, frameIndex, timestamp );
    }
    return 0;
}
