namespace libcgt { namespace core { namespace math {

uint16_t byteSwap16( uint16_t x )
{
    return ( x >> 8 ) | ( x << 8 );
}

uint32_t byteSwap16x2( uint32_t x )
{
    return
        ( ( x << 8 ) & 0xff00ff00 ) | // [ b2  0 b0  0 ]
        ( ( x >> 8 ) & 0x00ff00ff );  // [  0 b3  0 b1 ]
}

uint64_t byteSwap16x4( uint64_t x )
{
    return
        ( ( x << 8 ) & 0xff00ff00ff00ff00 ) | // [ b6  0 b4  0 b2  0 b0  0 ]
        ( ( x >> 8 ) & 0x00ff00ff00ff00ff );  // [  0 b7  0 b5  0 b3  0 b1 ]
}

uint32_t byteSwap32( uint32_t x )
{
    return
        ( x >> 24 ) | // [  0  0  0 b3 ]
        ( ( x >> 8 ) & 0x0000ff00 ) | // [  0  0 b2  0 ]
        ( ( x << 8 ) & 0x00ff0000 ) | // [  0 b1  0  0 ]
        ( ( x << 24 ) & 0xff000000 );  // [ b0  0  0  0 ]
}

uint64_t byteSwap32x2( uint64_t x )
{
    return
        ( ( x >> 24 ) & 0x000000ff000000ff ) | // [          b7          b3 ]
        ( ( x >> 8 ) & 0x0000ff000000ff00 ) |  // [       b6          b2    ]
        ( ( x << 8 ) & 0x00ff000000ff0000 ) |  // [    b5          b1       ]
        ( ( x << 24 ) & 0xff000000ff000000 );  // [ b4          b0          ]
}


uint64_t byteSwap64( uint64_t x )
{
    return
        ( x >> 56 ) |                          // [  0  0  0  0  0  0  0 b7 ]
        ( ( x >> 40 ) & 0x000000000000ff00 ) | // [  0  0  0  0  0  0 b6  0 ]
        ( ( x >> 24 ) & 0x0000000000ff0000 ) | // [  0  0  0  0  0 b5  0  0 ]
        ( ( x >> 8 ) & 0x00000000ff000000 ) |  // [  0  0  0  0 b4  0  0  0 ]
        ( ( x << 8 ) & 0x000000ff00000000 ) |  // [  0  0  0 b3  0  0  0  0 ]
        ( ( x << 24 ) & 0x0000ff0000000000 ) | // [  0  0 b2  0  0  0  0  0 ]
        ( ( x << 40 ) & 0x00ff000000000000 ) | // [  0 b1  0  0  0  0  0  0 ]
        ( ( x << 56 ) & 0xff00000000000000 );  // [ b0  0  0  0  0  0  0  0 ]
}

bool byteSwap16( Array1DReadView< uint16_t > src,
    Array1DWriteView< uint16_t > dst )
{
    if( src.size() != dst.size() )
    {
        return false;
    }

    if( src.packed() && dst.packed() )
    {
        size_t nWords64 = src.size() / 4;
        Array1DReadView< uint64_t > src64( src.pointer(), nWords64 );
        Array1DWriteView< uint64_t > dst64( dst.pointer(), nWords64 );
        for( size_t i = 0; i < nWords64; ++i )
        {
            dst64[ i ] = byteSwap16x4( src64[ i ] );
        }

        // Input length may not be a multiple of 4.
        if( nWords64 * 4 < src.size() )
        {
            // Could in theory do a byteSwap32() and then a byteSwap16()
            // if there were 3 elements left, but it's totally not worth it.
            for( size_t i = nWords64 * 4; i < src.size(); ++i )
            {
                dst[ i ] = byteSwap16( src[ i ] );
            }
        }
    }
    else
    {
        for( int i = 0; i < src.size(); ++i )
        {
            dst[ i ] = byteSwap16( src[ i ] );
        }
    }

    return true;
}

uint32_t mortonPack2D( uint16_t x, uint16_t y )
{
    static const unsigned int B[] =
        { 0x55555555, 0x33333333, 0x0f0f0f0f, 0x00ff00ff };
    static const unsigned int S[] = { 1, 2, 4, 8 };

    // Interleave lower 16 bits of x and y, so the bits of x
    // are in the even positions and bits from y in the odd.
    // z gets the resulting 32-bit Morton Number.
    uint32_t x32 = x;
    uint32_t y32 = y;
    uint32_t index;

    x32 = ( x32 | ( x32 << S[ 3 ] ) ) & B[ 3 ];
    x32 = ( x32 | ( x32 << S[ 2 ] ) ) & B[ 2 ];
    x32 = ( x32 | ( x32 << S[ 1 ] ) ) & B[ 1 ];
    x32 = ( x32 | ( x32 << S[ 0 ] ) ) & B[ 0 ];

    y32 = ( y32 | ( y32 << S[ 3 ] ) ) & B[ 3 ];
    y32 = ( y32 | ( y32 << S[ 2 ] ) ) & B[ 2 ];
    y32 = ( y32 | ( y32 << S[ 1 ] ) ) & B[ 1 ];
    y32 = ( y32 | ( y32 << S[ 0 ] ) ) & B[ 0 ];

    index = x32 | ( y32 << 1 );

    return index;
}

uint16x2 mortonUnpack2D( uint32_t index )
{
    uint64_t index64 = index;

    // Pack into 64-bits: [y | x].
    // 0xAAAAAAAA extracts odd bits.
    // 0x55555555 extracts even bits.
    // Only shift y by 31 since we want the bottom bit (which was the 1st, not
    // the 0th) to be in bit w[32].
    uint64_t w = ( ( index64 & 0xAAAAAAAA ) << 31 ) | ( index64 & 0x55555555 );

    w = ( w | ( w >> 1 ) ) & 0x3333333333333333;
    w = ( w | ( w >> 2 ) ) & 0x0f0f0f0f0f0f0f0f;
    w = ( w | ( w >> 4 ) ) & 0x00ff00ff00ff00ff;
    w = ( w | ( w >> 8 ) ) & 0x0000ffff0000ffff;

    uint16_t x = w & 0x000000000000ffff;
    uint16_t y = static_cast< uint16_t >( ( w & 0x0000ffff00000000 ) >> 32 );
    return{ x, y };
}

uint16_t mortonPack3D_5bit( uint8_t x, uint8_t y, uint8_t z )
{
    uint32_t index0 = x;
    uint32_t index1 = y;
    uint32_t index2 = z;

    index0 &= 0x0000001f;
    index1 &= 0x0000001f;
    index2 &= 0x0000001f;
    index0 *= 0x01041041;
    index1 *= 0x01041041;
    index2 *= 0x01041041;
    index0 &= 0x10204081;
    index1 &= 0x10204081;
    index2 &= 0x10204081;
    index0 *= 0x00011111;
    index1 *= 0x00011111;
    index2 *= 0x00011111;
    index0 &= 0x12490000;
    index1 &= 0x12490000;
    index2 &= 0x12490000;

    return static_cast< uint16_t >(
        ( index0 >> 16 ) | ( index1 >> 15 ) | ( index2 >> 14 ) );
}

uint8x3 mortonUnpack3D_5bit( uint16_t index )
{
    uint32_t value0 = index;
    uint32_t value1 = ( value0 >> 1 );
    uint32_t value2 = ( value0 >> 2 );

    value0 &= 0x00001249;
    value1 &= 0x00001249;
    value2 &= 0x00001249;
    value0 |= ( value0 >> 2 );
    value1 |= ( value1 >> 2 );
    value2 |= ( value2 >> 2 );
    value0 &= 0x000010c3;
    value1 &= 0x000010c3;
    value2 &= 0x000010c3;
    value0 |= ( value0 >> 4 );
    value1 |= ( value1 >> 4 );
    value2 |= ( value2 >> 4 );
    value0 &= 0x0000100f;
    value1 &= 0x0000100f;
    value2 &= 0x0000100f;
    value0 |= ( value0 >> 8 );
    value1 |= ( value1 >> 8 );
    value2 |= ( value2 >> 8 );
    value0 &= 0x0000001f;
    value1 &= 0x0000001f;
    value2 &= 0x0000001f;

    uint8_t x = static_cast< uint8_t >( value0 );
    uint8_t y = static_cast< uint8_t >( value1 );
    uint8_t z = static_cast< uint8_t >( value2 );
    return{ x, y, z };
}

uint32_t mortonPack3D_10bit( uint16_t x, uint16_t y, uint16_t z )
{
    uint32_t index0 = x;
    uint32_t index1 = y;
    uint32_t index2 = z;

    index0 &= 0x000003ff;
    index1 &= 0x000003ff;
    index2 &= 0x000003ff;
    index0 |= ( index0 << 16 );
    index1 |= ( index1 << 16 );
    index2 |= ( index2 << 16 );
    index0 &= 0x030000ff;
    index1 &= 0x030000ff;
    index2 &= 0x030000ff;
    index0 |= ( index0 << 8 );
    index1 |= ( index1 << 8 );
    index2 |= ( index2 << 8 );
    index0 &= 0x0300f00f;
    index1 &= 0x0300f00f;
    index2 &= 0x0300f00f;
    index0 |= ( index0 << 4 );
    index1 |= ( index1 << 4 );
    index2 |= ( index2 << 4 );
    index0 &= 0x030c30c3;
    index1 &= 0x030c30c3;
    index2 &= 0x030c30c3;
    index0 |= ( index0 << 2 );
    index1 |= ( index1 << 2 );
    index2 |= ( index2 << 2 );
    index0 &= 0x09249249;
    index1 &= 0x09249249;
    index2 &= 0x09249249;

    return( index0 | ( index1 << 1 ) | ( index2 << 2 ) );
}

uint16x3 mortonUnpack3D_10bit( uint32_t index )
{
    uint32_t value0 = index;
    uint32_t value1 = ( value0 >> 1 );
    uint32_t value2 = ( value0 >> 2 );

    value0 &= 0x09249249;
    value1 &= 0x09249249;
    value2 &= 0x09249249;
    value0 |= ( value0 >> 2 );
    value1 |= ( value1 >> 2 );
    value2 |= ( value2 >> 2 );
    value0 &= 0x030c30c3;
    value1 &= 0x030c30c3;
    value2 &= 0x030c30c3;
    value0 |= ( value0 >> 4 );
    value1 |= ( value1 >> 4 );
    value2 |= ( value2 >> 4 );
    value0 &= 0x0300f00f;
    value1 &= 0x0300f00f;
    value2 &= 0x0300f00f;
    value0 |= ( value0 >> 8 );
    value1 |= ( value1 >> 8 );
    value2 |= ( value2 >> 8 );
    value0 &= 0x030000ff;
    value1 &= 0x030000ff;
    value2 &= 0x030000ff;
    value0 |= ( value0 >> 16 );
    value1 |= ( value1 >> 16 );
    value2 |= ( value2 >> 16 );
    value0 &= 0x000003ff;
    value1 &= 0x000003ff;
    value2 &= 0x000003ff;

    uint16_t x = value0;
    uint16_t y = value1;
    uint16_t z = value2;

    return{ x, y, z };
}

} } } // math, core, libcgt

// TODO:
/*
// for 32-bit architectures
// morton1 - extract even bits

uint32_t morton1(uint32_t x)
{
x = x & 0x55555555;
x = (x | (x >> 1)) & 0x33333333;
x = (x | (x >> 2)) & 0x0F0F0F0F;
x = (x | (x >> 4)) & 0x00FF00FF;
x = (x | (x >> 8)) & 0x0000FFFF;
return x;
}

// morton2 - extract odd and even bits

void morton2(uint32_t *x, uint32_t *y, uint32_t z)
{
*x = morton1(z);
*y = morton1(z >> 1);
}
*/
