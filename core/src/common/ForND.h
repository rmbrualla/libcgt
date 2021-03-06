#pragma once

#include <string>

#include <vecmath/Vector2i.h>
#include <vecmath/Vector3i.h>
#include "ProgressReporter.h"

namespace libcgt { namespace core {

template< typename Func >
void for2D( const Vector2i& count, Func func );

template< typename Func >
void for2D( const Vector2i& count, const std::string& progressPrefix,
    Func func );

template< typename Func >
void for2D( const Vector2i& first, const Vector2i& count,
    Func func );

template< typename Func >
void for2D( const Vector2i& first, const Vector2i& count,
    const Vector2i& step, Func func );

template< typename Func >
void for2D( const Vector2i& first,
    const Vector2i& count, const Vector2i& step,
    const std::string& progressPrefix, Func func );

template< typename Func >
void for3D( const Vector3i& count, Func func );

template< typename Func >
void for3D( const Vector3i& count,
    const std::string& progressPrefix, Func func );

template< typename Func >
void for3D( const Vector3i& first,
    const Vector3i& count, Func func );

template< typename Func >
void for3D( const Vector3i& first, const Vector3i& count,
    const Vector3i& step, Func func );

template< typename Func >
void for3D( const Vector3i& first, const Vector3i& count,
    const Vector3i& step, const std::string& progressPrefix,
    Func func );

} } // core, libcgt

#include "ForND.inl"
