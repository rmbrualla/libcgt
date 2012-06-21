#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <QString>

#include "math/Arithmetic.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4d.h"
#include "vecmath/Vector4i.h"

Vector4f::Vector4f()
{
	m_elements[0] = 0;
	m_elements[1] = 0;
	m_elements[2] = 0;
	m_elements[3] = 0;
}

// TODO: do the same for vec2 and 3 and double
// TODO: switch to constructor initializer
Vector4f::Vector4f( float f )
{
	m_elements[ 0 ] = f;
	m_elements[ 1 ] = f;
	m_elements[ 2 ] = f;
	m_elements[ 3 ] = f;
}

Vector4f::Vector4f( float fx, float fy, float fz, float fw )
{
	m_elements[0] = fx;
	m_elements[1] = fy;
	m_elements[2] = fz;
	m_elements[3] = fw;
}

Vector4f::Vector4f( float buffer[ 4 ] )
{
	m_elements[ 0 ] = buffer[ 0 ];
	m_elements[ 1 ] = buffer[ 1 ];
	m_elements[ 2 ] = buffer[ 2 ];
	m_elements[ 3 ] = buffer[ 3 ];
}

Vector4f::Vector4f( const Vector2f& xy, float z, float w )
{
	m_elements[0] = xy.x;
	m_elements[1] = xy.y;
	m_elements[2] = z;
	m_elements[3] = w;
}

Vector4f::Vector4f( float x, const Vector2f& yz, float w )
{
	m_elements[0] = x;
	m_elements[1] = yz.x;
	m_elements[2] = yz.y;
	m_elements[3] = w;
}

Vector4f::Vector4f( float x, float y, const Vector2f& zw )
{
	m_elements[0] = x;
	m_elements[1] = y;
	m_elements[2] = zw.x;
	m_elements[3] = zw.y;
}

Vector4f::Vector4f( const Vector2f& xy, const Vector2f& zw )
{
	m_elements[0] = xy.x;
	m_elements[1] = xy.y;
	m_elements[2] = zw.x;
	m_elements[3] = zw.y;
}

Vector4f::Vector4f( const Vector3f& xyz, float w )
{
	m_elements[0] = xyz.x;
	m_elements[1] = xyz.y;
	m_elements[2] = xyz.z;
	m_elements[3] = w;
}

Vector4f::Vector4f( float x, const Vector3f& yzw )
{
	m_elements[0] = x;
	m_elements[1] = yzw.x;
	m_elements[2] = yzw.y;
	m_elements[3] = yzw.z;
}

Vector4f::Vector4f( const Vector4f& rv )
{
	m_elements[0] = rv.m_elements[0];
	m_elements[1] = rv.m_elements[1];
	m_elements[2] = rv.m_elements[2];
	m_elements[3] = rv.m_elements[3];
}

Vector4f::Vector4f( const Vector4d& rv )
{
	m_elements[ 0 ] = static_cast< float >( rv.x );
	m_elements[ 1 ] = static_cast< float >( rv.y );
	m_elements[ 2 ] = static_cast< float >( rv.z );
	m_elements[ 3 ] = static_cast< float >( rv.w );
}

Vector4f::Vector4f( const Vector4i& rv )
{
	m_elements[ 0 ] = static_cast< float >( rv.x );
	m_elements[ 1 ] = static_cast< float >( rv.y );
	m_elements[ 2 ] = static_cast< float >( rv.z );
	m_elements[ 3 ] = static_cast< float >( rv.w );
}

Vector4f& Vector4f::operator = ( const Vector4f& rv )
{
	if( this != &rv )
	{
		m_elements[0] = rv.m_elements[0];
		m_elements[1] = rv.m_elements[1];
		m_elements[2] = rv.m_elements[2];
		m_elements[3] = rv.m_elements[3];
	}
	return *this;
}

const float& Vector4f::operator [] ( int i ) const
{
	return m_elements[ i ];
}

float& Vector4f::operator [] ( int i )
{
	return m_elements[ i ];
}

Vector2f Vector4f::xy() const
{
	return Vector2f( x, y );
}

Vector2f Vector4f::yz() const
{
	return Vector2f( y, z );
}

Vector2f Vector4f::zw() const
{
	return Vector2f( z, w );
}

Vector2f Vector4f::wx() const
{
	return Vector2f( w, x );
}

Vector3f Vector4f::xyz() const
{
	return Vector3f( x, y, z );
}

Vector3f Vector4f::yzw() const
{
	return Vector3f( y, z, w );
}

Vector3f Vector4f::zwx() const
{
	return Vector3f( z, w, x );
}

Vector3f Vector4f::wxy() const
{
	return Vector3f( w, x, y );
}

Vector3f Vector4f::xyw() const
{
	return Vector3f( x, y, w );
}

Vector3f Vector4f::yzx() const
{
	return Vector3f( y, z, x );
}

Vector3f Vector4f::zwy() const
{
	return Vector3f( z, w, y );
}

Vector3f Vector4f::wxz() const
{
	return Vector3f( w, x, z );
}

float Vector4f::abs() const
{
	return norm();
}

float Vector4f::absSquared() const
{
	return normSquared();
}

float Vector4f::norm() const
{
	return sqrt( normSquared() );
}

float Vector4f::normSquared() const
{
	return x * x + y * y + z * z + w * w;
}

void Vector4f::normalize()
{
	float rcpNorm = 1.0f / norm();
	x *= rcpNorm;
	y *= rcpNorm;
	z *= rcpNorm;
	w *= rcpNorm;
}

Vector4f Vector4f::normalized() const
{
	float rcpNorm = 1.0f / norm();
	return Vector4f
	(
		x * rcpNorm, 
		y * rcpNorm, 
		z * rcpNorm, 
		w * rcpNorm
	);
}

void Vector4f::homogenize()
{
	if( w != 0 )
	{
		float rcpW = 1.0f / w;
		x *= rcpW;
		y *= rcpW;
		z *= rcpW;
		w = 1;
	}
}

Vector4f Vector4f::homogenized() const
{
	if( w != 0 )
	{
		float rcpW = 1.0f / w;
		return Vector4f( rcpW * x, rcpW * y, rcpW * z, 1 );
	}
	else
	{
		return Vector4f( x, y, z, w );
	}
}

void Vector4f::negate()
{
	x = -x;
	y = -y;
	z = -z;
	w = -w;
}

Vector4i Vector4f::floored() const
{
	return Vector4i
	(
		Arithmetic::floorToInt( x ),
		Arithmetic::floorToInt( y ),
		Arithmetic::floorToInt( z ),
		Arithmetic::floorToInt( w )
	);
}

Vector4f::operator const float* () const
{
	return m_elements;
}

Vector4f::operator float* ()
{
	return m_elements;
}

QString Vector4f::toString() const
{
	QString out;

	out.append( "( " );
	out.append( QString( "%1" ).arg( x, 10, 'g', 4 ) );
	out.append( QString( "%1" ).arg( y, 10, 'g', 4 ) );
	out.append( QString( "%1" ).arg( z, 10, 'g', 4 ) );
	out.append( QString( "%1" ).arg( w, 10, 'g', 4 ) );
	out.append( " )" );

	return out;
}

// static
float Vector4f::dot( const Vector4f& v0, const Vector4f& v1 )
{
	return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}

// static
Vector4f Vector4f::lerp( const Vector4f& v0, const Vector4f& v1, float alpha )
{
	return alpha * ( v1 - v0 ) + v0;
}
