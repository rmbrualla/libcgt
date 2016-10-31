#pragma once

#include <cassert>

#include <cuda_runtime.h>

#include <common/Array2DView.h>

// TODO: create an Interop version.
// TODO: support CUDA surfaces: bindSurface()
// TODO: use cudaMalloc3DArray() for an ND array.

// Wrapper around CUDA "array" memory.
// Allocates 2D memory using cudaMallocArray(), which can only be used as
// textures or surfaces.
//
// T should be a CUDA element type such uchar, schar2, or float4.
template< typename T >
class DeviceOpaqueArray2D
{
public:

    DeviceOpaqueArray2D() = default;
    DeviceOpaqueArray2D( const Vector2i& size );

    // Wrap an existing cuArray in a DeviceOpaqueArray2D object *without*
    // taking ownership.
    DeviceOpaqueArray2D( cudaArray_t array );

    ~DeviceOpaqueArray2D();
    // TODO: move constructors and assignment operator.

    bool isNull() const;
    bool notNull() const;

    const cudaChannelFormatDesc& channelFormatDescription() const;
    const cudaResourceDesc& resourceDescription() const;

    int width() const;
    int height() const;
    Vector2i size() const;
    int numElements() const;

    // For the copy to succeed, sizes must be exact:
    //   src.size() == size() - dstOffset
    // In addition, src.elementsArePacked() or src.packed() must be true.
    bool copyFromHost( Array2DView< const T > src,
        const Vector2i& dstOffset = Vector2i{ 0 } );

    // For the copy to succeed, sizes must be exact:
    //   size() - dstOffset == dst.size()
    // In addition, dst.elementsArePacked() or dst.packed() must be true.
    bool copyToHost( Array2DView< T > dst,
        const Vector2i& srcOffset = Vector2i{ 0 } ) const;

    const cudaArray_t deviceArray() const;
    cudaArray_t deviceArray();

private:

    // TODO: Vector2< size_t >.
    Vector2i m_size = Vector2i{ 0 };
    cudaChannelFormatDesc m_cfd = {};
    cudaResourceDesc m_resourceDesc = {};
    cudaArray_t m_deviceArray = nullptr;
    bool m_ownsArray = true;

};

#include "DeviceOpaqueArray2D.inl"
