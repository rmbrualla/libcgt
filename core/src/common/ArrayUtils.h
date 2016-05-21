#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "Array1DView.h"
#include "Array2D.h"
#include "Array3D.h"
#include "BasicTypes.h" // TODO: uint8x4 --> Vector4ub

#include <vecmath/Box3i.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>

namespace libcgt { namespace core { namespace arrayutils {

// Copy between two Array1DViews, with potentially varying stride.
// If both are packed(), uses memcpy to do a single fast copy.
// Otherwise, iterates the copy one element at a time.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array1DView< const T > src, Array1DView< T > dst );

// Copy between two Array2DViews, with potentially varying strides.
// If both are packed(), uses memcpy to do a single fast copy.
// If both row strides are the same, then uses memcpy row by row.
// Otherwise, iterates the copy one element at a time.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array2DView< const T > src, Array2DView< T > dst );

// TODO(jiawen): copy of Array3DView.

// TODO: rename this to sliceChannel()?
// Given an existing Array1DView< T >, returns a Array1DView< S > with the
// same stride, but with elements of type S where S is a component of T,
// and is at offset "componentOffsetBytes" within T.
template< typename S, typename T >
Array1DView< S > componentView( Array1DView< T > src, int componentOffsetBytes );

// Given an existing Array2DView< T >, returns a Array2DView< S > with the
// same stride, but with elements of type S where S is a component of T,
// and is at offset "componentOffsetBytes" within T.
template< typename S, typename T >
Array2DView< S > componentView( Array2DView< T > src, int componentOffsetBytes );

// TODO(jiawen): crop of Array1DView.

// Get a rectangular subset of an Array2DView, starting at xy.
template< typename T >
Array2DView< T > crop( Array2DView< T > view, const Vector2i& xy );

// Get a rectangular subset of an Array2DView.
template< typename T >
Array2DView< T > crop( Array2DView< T > view, const Rect2i& rect );

// Get a box subset of a Array3DView, starting at xyz.
template< typename T >
Array3DView< T > crop( Array3DView< T > view, const Vector3i& xyz );

// Get a box subset of an Array3DView.
template< typename T >
Array3DView< T > crop( Array3DView< T > view, const Box3i& box );

// TODO: implement clear() using fill(), specialize fill() on uint8_t
// use specialized view if isPacked() and T == 0?
template< typename T >
bool clear( Array2DView< T > view );

template< typename T >
bool fill( Array1DView< T > view, const T& value );

template< typename T >
bool fill( Array2DView< T > view, const T& value );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array1DView< T > flipLeftRight(Array1DView< T > src);

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array2DView< T > flipLeftRight( Array2DView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array2DView< T > flipUpDown( Array2DView< T > src );

// TODO(jiawen): tranpose on 2D.


// Classical "map" function: dst[ i ] = f( src[ i ] )
#if 0
// TODO(MSVC_2015): the following should work but does not in VS2013.
template< typename TSrc, typename TDst >
static bool map( Array1DView< TSrc > src, Array1DView< TDst > dst,
    std::function< TDst( TSrc ) > f );
#else
// f should be a function object that mapping TSrc -> TDst
template< typename TSrc, typename TDst, typename Func >
bool map( Array1DView< TSrc > src, Array1DView< TDst > dst, Func f );
#endif

} } } // arrayutils, core, libcgt

class ArrayUtils
{
public:

    // TODO: saveTMP
    // TODO: vector --> Array1DView<T>
    template< typename T >
    static bool loadBinary( FILE* fp, std::vector< T >& output );

    template< typename T >
    static bool saveBinary( const std::vector< T >& input, const char* filename );

    template< typename T >
    static bool saveBinary( const std::vector< T >& input, FILE* fp );

    static bool saveTXT( Array1DView< const int16_t > view, const char* filename );
    static bool saveTXT( Array1DView< const int32_t > view, const char* filename );

    static bool saveTXT( Array1DView< const float >& view, const char* filename );
    static bool saveTXT( Array1DView< const Vector2f >& view, const char* filename );
    static bool saveTXT( Array1DView< const Vector3f >& view, const char* filename );
    static bool saveTXT( Array1DView< const Vector4f >& view, const char* filename );

    static bool saveTXT( Array2DView< const uint8_t > view, const char* filename );
    static bool saveTXT( Array2DView< const uint8x4 > view, const char* filename );

    static bool saveTXT( Array2DView< const int16_t > view, const char* filename );

    static bool saveTXT( Array2DView< const float > view, const char* filename );
    static bool saveTXT( Array2DView< const Vector4f > view, const char* filename );

    static bool saveTXT( Array3DView< const float > view, const char* filename );
    static bool saveTXT( Array3DView< const Vector2f > view, const char* filename );
};

#include "common/ArrayUtils.inl"
