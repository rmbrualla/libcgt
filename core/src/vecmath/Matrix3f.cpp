#include "vecmath/Matrix3f.h"

#include <cassert>
#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>

#include <math/MathUtils.h>

#include "Matrix2f.h"
#include "Quat4f.h"
#include "Vector2f.h"

using std::abs;

Matrix3f::Matrix3f() :
    Matrix3f( 0.0f )
{

}

Matrix3f::Matrix3f( float fill )
{
    for( int i = 0; i < 9; ++i )
    {
        m_elements[ i ] = fill;
    }
}

Matrix3f::Matrix3f( float m00, float m01, float m02,
                   float m10, float m11, float m12,
                   float m20, float m21, float m22 )
{
    m_elements[ 0 ] = m00;
    m_elements[ 1 ] = m10;
    m_elements[ 2 ] = m20;

    m_elements[ 3 ] = m01;
    m_elements[ 4 ] = m11;
    m_elements[ 5 ] = m21;

    m_elements[ 6 ] = m02;
    m_elements[ 7 ] = m12;
    m_elements[ 8 ] = m22;
}

Matrix3f::Matrix3f( const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, bool setColumns )
{
    if( setColumns )
    {
        setCol( 0, v0 );
        setCol( 1, v1 );
        setCol( 2, v2 );
    }
    else
    {
        setRow( 0, v0 );
        setRow( 1, v1 );
        setRow( 2, v2 );
    }
}

const float& Matrix3f::operator () ( int i, int j ) const
{
    return m_elements[ j * 3 + i ];
}

float& Matrix3f::operator () ( int i, int j )
{
    return m_elements[ j * 3 + i ];
}

Vector3f Matrix3f::getRow( int i ) const
{
    return Vector3f
    (
        m_elements[ i ],
        m_elements[ i + 3 ],
        m_elements[ i + 6 ]
    );
}

void Matrix3f::setRow( int i, const Vector3f& v )
{
    m_elements[ i ] = v.x;
    m_elements[ i + 3 ] = v.y;
    m_elements[ i + 6 ] = v.z;
}

Vector3f Matrix3f::getCol( int j ) const
{
    int colStart = 3 * j;

    return Vector3f
    (
        m_elements[ colStart ],
        m_elements[ colStart + 1 ],
        m_elements[ colStart + 2 ]
    );
}

void Matrix3f::setCol( int j, const Vector3f& v )
{
    int colStart = 3 * j;

    m_elements[ colStart ] = v.x;
    m_elements[ colStart + 1 ] = v.y;
    m_elements[ colStart + 2 ] = v.z;
}

Matrix2f Matrix3f::getSubmatrix2x2( int i0, int j0 ) const
{
    Matrix2f out;

    for( int i = 0; i < 2; ++i )
    {
        for( int j = 0; j < 2; ++j )
        {
            out( i, j ) = ( *this )( i + i0, j + j0 );
        }
    }

    return out;
}

void Matrix3f::setSubmatrix2x2( int i0, int j0, const Matrix2f& m )
{
    for( int i = 0; i < 2; ++i )
    {
        for( int j = 0; j < 2; ++j )
        {
            ( *this )( i + i0, j + j0 ) = m( i, j );
        }
    }
}

float Matrix3f::determinant() const
{
    return Matrix3f::determinant3x3
    (
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22
    );
}

Matrix3f Matrix3f::inverse() const
{
    bool isSingular;
    return inverse( isSingular );
}

Matrix3f Matrix3f::inverse( bool& isSingular, float epsilon ) const
{
    float cofactor00 =  Matrix2f::determinant2x2( m11, m12, m21, m22 );
    float cofactor01 = -Matrix2f::determinant2x2( m10, m12, m20, m22 );
    float cofactor02 =  Matrix2f::determinant2x2( m10, m11, m20, m21 );

    float cofactor10 = -Matrix2f::determinant2x2( m01, m02, m21, m22 );
    float cofactor11 =  Matrix2f::determinant2x2( m00, m02, m20, m22 );
    float cofactor12 = -Matrix2f::determinant2x2( m00, m01, m20, m21 );

    float cofactor20 =  Matrix2f::determinant2x2( m01, m02, m11, m12 );
    float cofactor21 = -Matrix2f::determinant2x2( m00, m02, m10, m12 );
    float cofactor22 =  Matrix2f::determinant2x2( m00, m01, m10, m11 );

    float determinant = m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02;

    isSingular = ( abs( determinant ) < epsilon );
    if( isSingular )
    {
        return Matrix3f();
    }
    else
    {
        float reciprocalDeterminant = 1.0f / determinant;

        return Matrix3f
        (
            cofactor00 * reciprocalDeterminant, cofactor10 * reciprocalDeterminant, cofactor20 * reciprocalDeterminant,
            cofactor01 * reciprocalDeterminant, cofactor11 * reciprocalDeterminant, cofactor21 * reciprocalDeterminant,
            cofactor02 * reciprocalDeterminant, cofactor12 * reciprocalDeterminant, cofactor22 * reciprocalDeterminant
        );
    }
}

void Matrix3f::transpose()
{
    float temp;

    for( int i = 0; i < 2; ++i )
    {
        for( int j = i + 1; j < 3; ++j )
        {
            temp = ( *this )( i, j );
            ( *this )( i, j ) = ( *this )( j, i );
            ( *this )( j, i ) = temp;
        }
    }
}

Matrix3f Matrix3f::transposed() const
{
    Matrix3f out;
    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            out( j, i ) = ( *this )( i, j );
        }
    }

    return out;
}

Matrix3f::operator const float* () const
{
    return m_elements;
}

Matrix3f::operator float* ()
{
    return m_elements;
}

std::string Matrix3f::toString() const
{
    const int FIELD_WIDTH = 8;
    const int PRECISION = 4;

    std::ostringstream sstream;
    sstream << std::fixed << std::setprecision( PRECISION ) <<
        std::setiosflags( std::ios::right );
    sstream << std::endl;

    for( int i = 0; i < 3; ++i )
    {
        sstream << "[ ";
        for( int j = 0; j < 3; ++j )
        {
            sstream << std::setw( FIELD_WIDTH ) << ( *this )( i, j ) << " ";
        }
        sstream << "]" << std::endl;
    }
    return sstream.str();
}

Vector2f Matrix3f::transformPoint( const Vector2f& p ) const
{
    return ( ( *this ) * Vector3f( p, 1 ) ).xy;
}

Vector2f Matrix3f::transformVector( const Vector2f& v ) const
{
    return ( ( *this ) * Vector3f( v, 0 ) ).xy;
}

// static
float Matrix3f::determinant3x3( float m00, float m01, float m02,
                               float m10, float m11, float m12,
                               float m20, float m21, float m22 )
{
    return
    (
            m00 * ( m11 * m22 - m12 * m21 )
        - m01 * ( m10 * m22 - m12 * m20 )
        + m02 * ( m10 * m21 - m11 * m20 )
    );
}

// static
Matrix3f Matrix3f::ones()
{
    return Matrix3f( 1.0f );
}

// static
Matrix3f Matrix3f::identity()
{
    Matrix3f m;

    m( 0, 0 ) = 1;
    m( 1, 1 ) = 1;
    m( 2, 2 ) = 1;

    return m;
}

// static
Matrix3f Matrix3f::rotateX( float radians )
{
    float c = cos( radians );
    float s = sin( radians );

    return Matrix3f
    (
        1, 0, 0,
        0, c, -s,
        0, s, c
    );
}

// static
Matrix3f Matrix3f::rotateY( float radians )
{
    float c = cos( radians );
    float s = sin( radians );

    return Matrix3f
    (
        c, 0, s,
        0, 1, 0,
        -s, 0, c
    );
}

// static
Matrix3f Matrix3f::rotateZ( float radians )
{
    float c = cos( radians );
    float s = sin( radians );

    return Matrix3f
    (
        c, -s, 0,
        s, c, 0,
        0, 0, 1
    );
}

// static
Matrix3f Matrix3f::rotation( const Vector3f& axis, float radians )
{
    float c = cos( radians );
    float s = sin( radians );
    float omc = 1.0f - c;

    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    return Matrix3f
    (
        x * x * omc + c, y * x * omc - z * s, z * x * omc + y * s,
        x * y * omc + z * s, y * y * omc + c, z * y * omc - x * s,
        x * z * omc - y * s, y * z * omc + x * s, z * z * omc + c
    );
}

// static
Matrix3f Matrix3f::rotation( const Vector3f& axisAngle )
{
    float radians;
    Vector3f axis = axisAngle.normalized( radians );
    return rotation( axis, radians );
}

// static
Matrix3f Matrix3f::scaling( const Vector3f& xyz )
{
    return Matrix3f
    (
        xyz.x, 0, 0,
        0, xyz.y, 0,
        0, 0, xyz.z
    );
}

// static
Matrix3f Matrix3f::uniformScaling( float s )
{
    return Matrix3f
    (
        s, 0, 0,
        0, s, 0,
        0, 0, s
    );
}

// static
Matrix3f Matrix3f::translation( const Vector2f& xy )
{
    return Matrix3f
    (
        1, 0, xy.x,
        0, 1, xy.y,
        0, 0, 1
    );
}

// static
Matrix3f Matrix3f::scaleTranslate( const Vector2f& srcOrigin, const Vector2f& srcSize,
    const Vector2f& dstOrigin, const Vector2f& dstSize )
{
    // translate rectangle to have its origin at (0,0)
    Matrix3f t0 = Matrix3f::translation( -srcOrigin );

    // scale it to [0,1]^2, then to [0,dstSize]
    Matrix3f s = Matrix3f::scaling( { dstSize.x / srcSize.x, dstSize.y / srcSize.y, 1 } );

    // translate rectangle to dstOrigin
    Matrix3f t1 = Matrix3f::translation( dstOrigin );

    return t1 * s * t0;
}

// static
Matrix3f Matrix3f::fromQuat( const Quat4f& q )
{
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;

    float xy = q.x * q.y;
    float zw = q.z * q.w;

    float xz = q.x * q.z;
    float yw = q.y * q.w;

    float yz = q.y * q.z;
    float xw = q.x * q.w;

    return Matrix3f
    (
        1.0f - 2.0f * ( yy + zz ), 2.0f * ( xy - zw ), 2.0f * ( xz + yw ),
        2.0f * ( xy + zw ), 1.0f - 2.0f * ( xx + zz ), 2.0f * ( yz - xw ),
        2.0f * ( xz - yw ), 2.0f * ( yz + xw ), 1.0f - 2.0f * ( xx + yy )
    );
}

// static
Matrix3f Matrix3f::randomRotation( float u0, float u1, float u2 )
{
    return Matrix3f::fromQuat( Quat4f::randomRotation( u0, u1, u2 ) );
}

Matrix3f operator + ( const Matrix3f& x, const Matrix3f& y )
{
    Matrix3f sum;

    for( int k = 0; k < 9; ++k )
    {
        sum[k] = x[k] + y[k];
    }

    return sum;
}

Matrix3f operator - ( const Matrix3f& x, const Matrix3f& y )
{
    Matrix3f difference;

    for( int k = 0; k < 9; ++k )
    {
        difference[k] = x[k] - y[k];
    }

    return difference;
}

Matrix3f operator - ( const Matrix3f& x )
{
    Matrix3f output;

    for( int k = 0; k < 9; ++k )
    {
        output[k] = -x[k];
    }

    return output;
}

Matrix3f operator * ( float f, const Matrix3f& m )
{
    Matrix3f output;

    for( int k = 0; k < 9; ++k )
    {
        output[k] = f * m[k];
    }

    return output;
}

Matrix3f operator * ( const Matrix3f& m, float f )
{
    return f * m;
}

Matrix3f operator / ( const Matrix3f& m, float f )
{
    return ( 1.0f / f ) * m;
}

Vector3f operator * ( const Matrix3f& m, const Vector3f& v )
{
    Vector3f output( 0, 0, 0 );

    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            output[ i ] += m( i, j ) * v[ j ];
        }
    }

    return output;
}

Matrix3f operator * ( const Matrix3f& x, const Matrix3f& y )
{
    Matrix3f product; // zeroes

    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            for( int k = 0; k < 3; ++k )
            {
                product( i, k ) += x( i, j ) * y( j, k );
            }
        }
    }

    return product;
}

#if 0
// TODO: port to a new module

// static
void GMatrixd::homography( QVector< Vector3f > from,
    QVector< Vector3f > to, GMatrixd& output )
{
    output.resize( 3, 3 );
    GMatrixd m( 8, 9 );

    for( int i = 0; i < 4; ++i )
    {
        m(2*i,0) = from[i].x() * to[i].z();
        m(2*i,1) = from[i].y() * to[i].z();
        m(2*i,2) = from[i].z() * to[i].z();
        m(2*i,6) = -from[i].x() * to[i].x();
        m(2*i,7) = -from[i].y() * to[i].x();
        m(2*i,8) = -from[i].z() * to[i].x();

        m(2*i+1,3) = from[i].x() * to[i].z();
        m(2*i+1,4) = from[i].y() * to[i].z();
        m(2*i+1,5) = from[i].z() * to[i].z();
        m(2*i+1,6) = -from[i].x() * to[i].y();
        m(2*i+1,7) = -from[i].y() * to[i].y();
        m(2*i+1,8) = -from[i].z() * to[i].y();
    }

    QVector< QVector< double > > eigenvectors;
    QVector< double > eigenvalues;

    GMatrixd mt( 9, 8 );
    m.transpose( mt );

    GMatrixd mtm( 9, 9 );
    GMatrixd::times( mt, m, mtm );

    mtm.eigenvalueDecomposition( &eigenvectors, &eigenvalues );

    for( int i=0;i<3;i++){
        for( int j=0;j<3;j++){
            output( i, j ) = eigenvectors[0][3*i+j];
        }
    }
}
#endif
