#include "geometry/GeometryUtils.h"

#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>

#include "common/ForND.h"
#include "geometry/Plane3f.h"
#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Box3f.h"
#include "vecmath/Matrix2f.h"
#include "vecmath/Matrix3f.h"
#include "vecmath/Matrix4f.h"
#include "vecmath/Rect2f.h"
#include "vecmath/Rect2i.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"

using libcgt::core::for2D;
using std::abs;
using std::max;
using std::min;
using std::swap;

// static
const float GeometryUtils::EPSILON = 0.0001f;

// static
Box3f GeometryUtils::boundingBoxForPoints( const std::vector< Vector3f >& points )
{
    if( points.size() == 0 )
    {
        return Box3f();
    }

    Vector3f minimum = points[0];
    Vector3f maximum = points[0];

    for( size_t i = 1; i < points.size(); ++i )
    {
        Vector3f xyz = points[i];
        float x = xyz.x;
        float y = xyz.y;
        float z = xyz.z;

        if( x < minimum.x )
        {
            minimum.x = x;
        }
        if( x > maximum.x )
        {
            maximum.x = x;
        }
        if( y < minimum.y )
        {
            minimum.y = y;
        }
        if( y > maximum.y )
        {
            maximum.y = y;
        }
        if( z < minimum.z )
        {
            minimum.z = z;
        }
        if( z > maximum.z )
        {
            maximum.z = z;
        }
    }

    return{ minimum, maximum - minimum };
}

// static
Rect2f GeometryUtils::triangleBoundingBox( const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    float minX = min( min( v0.x, v1.x ), v2.x );
    float minY = min( min( v0.y, v1.y ), v2.y );

    float maxX = max( max( v0.x, v1.x ), v2.x );
    float maxY = max( max( v0.y, v1.y ), v2.y );

    return{ { minX, minY }, { maxX - minX, maxY - minY } };
}

// static
Vector2f GeometryUtils::triangleCentroid( const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    return ( v0 + v1 + v2 ) / 3.0f;
}

// static
std::vector< Vector2f > GeometryUtils::pixelsInTriangle( const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    Rect2i bbox = triangleBoundingBox( v0, v1, v2 ).enlargedToInt();

    std::vector< Vector2f > pointsInside;
    pointsInside.reserve( bbox.area() );

    for2D( bbox.minimum(), bbox.size, [&]( const Vector2i& xy )
    {
        Vector2f p = xy + 0.5f;

        if( pointInTriangle( p, v0, v1, v2 ) )
        {
            pointsInside.push_back( p );
        }
    });

    return pointsInside;
}

// static
float GeometryUtils::edgeTest( const Vector2f& edgeNormal, const Vector2f& edgeOrigin, const Vector2f& point )
{
    return Vector2f::dot( edgeNormal, point - edgeOrigin );
}

// static
float GeometryUtils::edgeTestConservative( const Vector2f& edgeNormal, const Vector2f& edgeOrigin, const Vector2f& point )
{
    float tx = -0.5f;
    float ty = -0.5f;

    if( edgeNormal.x >= 0 )
    {
        tx = 0.5f;
    }
    if( edgeNormal.y >= 0 )
    {
        ty = 0.5f;
    }

    Vector2f offset{ tx, ty };

    return Vector2f::dot( edgeNormal, point ) + edgeTest( edgeNormal, edgeOrigin, offset );
}

// static
std::vector< Vector2f > GeometryUtils::pixelsInTriangleConservative(
    const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    // set up edges
    Vector2f edge01 = v1 - v0;
    Vector2f edge12 = v2 - v1;
    Vector2f edge20 = v0 - v2;

    // get edge normals
    Vector2f normal01 = edge01.normal();
    Vector2f normal12 = edge12.normal();
    Vector2f normal20 = edge20.normal();

    Rect2i bbox = triangleBoundingBox( v0, v1, v2 ).enlargedToInt();

    std::vector< Vector2f > pointsInside;
    pointsInside.reserve( bbox.area() );

    for2D( bbox.minimum(), bbox.size, [&]( const Vector2i& xy )
    {
        Vector2f p = xy + 0.5f;

        float passed01 = edgeTestConservative( normal01, v0, p );
        float passed12 = edgeTestConservative( normal12, v1, p );
        float passed20 = edgeTestConservative( normal20, v2, p );

        if( ( passed01 > 0 ) &&
            ( passed12 > 0 ) &&
            ( passed20 > 0 ) )
        {
            pointsInside.push_back( p );
        }
    });

    return pointsInside;
}

// static
bool GeometryUtils::pointsOnSameSide( const Vector2f& p0, const Vector2f& p1,
                                     const Vector2f& v0, const Vector2f& v1 )
{
    Vector2f edge = v1 - v0;
    Vector2f d0 = p0 - v0;
    Vector2f d1 = p1 - v0;

    Vector3f crossProduct0 = Vector2f::cross( edge, d0 );
    Vector3f crossProduct1 = Vector2f::cross( edge, d1 );

    return( Vector3f::dot( crossProduct0, crossProduct1 ) >= 0 );
}

// static
bool GeometryUtils::pointInTriangle( const Vector2f& point,
    const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    bool edge0 = pointsOnSameSide( point, v0, v1, v2 );
    bool edge1 = pointsOnSameSide( point, v1, v2, v0 );
    bool edge2 = pointsOnSameSide( point, v2, v0, v1 );

    return( edge0 && edge1 && edge2 );
}

// static
Vector3f GeometryUtils::euclideanToBarycentric( const Vector2f& p,
    const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    // r = l0 v0 + l1 v1 + l2 v2
    // this yields the linear system
    // [ ( x0 - x2 )    ( x1 - x2 ) ] [ l0 ] = [ x - x2 ]
    // [ ( y0 - y2 )    ( y1 - y2 ) ] [ l1 ]   [ y - y2 ]
    // a l = b

    Matrix2f a
    (
        v0.x - v2.x, v1.x - v2.x,
        v0.y - v2.y, v1.y - v2.y
    );

    Vector2f b
    {
        p.x - v2.x,
        p.y - v2.y
    };

    bool bSingular;
    Matrix2f ai = a.inverse( bSingular );
    assert( !bSingular );

    Vector2f l0l1 = ai * b;
    return Vector3f( l0l1, 1 - l0l1.x - l0l1.y );
}

// static
Vector2f GeometryUtils::barycentricToEuclidean( const Vector3f& b,
    const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    return( b.x * v0 + b.y * v1 + b.z * v2 );
}

// static
Vector3f GeometryUtils::barycentricToEuclidean( const Vector3f& b,
    const Vector3f& v0, const Vector3f& v1, const Vector3f& v2 )
{
    return( b.x * v0 + b.y * v1 + b.z * v2 );
}

// static
void GeometryUtils::getBasis( const Vector3f& n, Vector3f& b0, Vector3f& b1, float epsilon )
{
    // if n is tiny, just return x and y
    if( n.normSquared() < epsilon )
    {
        b0 = Vector3f( 1, 0, 0 );
        b1 = Vector3f( 0, 1, 0 );
        return;
    }

    // find an axis not parallel to n
    Vector3f axis;

    if( abs( n.x ) <= abs( n.y ) && abs( n.x ) <= abs( n.z ) )
    {
        axis = Vector3f( 1, 0, 0 );
    }
    else if( abs( n.y ) <= abs( n.z ) )
    {
        axis = Vector3f( 0, 1, 0 );
    }
    else
    {
        axis = Vector3f( 0, 0, 1 );
    }

    b0 = Vector3f::cross( n, axis ).normalized(); // first basis vector
    b1 = Vector3f::cross( n, b0 ).normalized(); // second basis vector
}

// static
Matrix3f GeometryUtils::getRightHandedBasis( const Vector3f& z )
{
    Vector3f x;
    Vector3f y;
    Vector3f z2 = z.normalized();

    getBasis( z2, x, y );

    Matrix3f m( x, y, z2 );

    if( m.determinant() < 0 )
    {
        return Matrix3f( y, x, z2 );
    }
    else
    {
        return m;
    }
}

// static
Matrix4f GeometryUtils::getRightHandedBasis( const Vector4f& z )
{
    Matrix3f m = getRightHandedBasis( z.xyz );
    Matrix4f output; // 0 matrix
    output.setSubmatrix3x3( 0, 0, m );
    output( 3, 3 ) = 1;
    return output;
}

// static
Matrix3f GeometryUtils::getRightHandedBasisWithPreferredUp( const Vector3f& z, const Vector3f& preferredY )
{
    Vector3f z2 = z.normalized();

    Vector3f x = Vector3f::cross( preferredY, z ).normalized();
    Vector3f y2 = Vector3f::cross( z2, x );

    return Matrix3f( x, y2, z2 );
}

// static
bool GeometryUtils::pointInsideSphere( const Vector3f& crPoint,
                                      const Vector3f& crSphereCenter, float sphereRadius )
{
    Vector3f diff = crPoint - crSphereCenter;
    return( diff.normSquared() < ( sphereRadius * sphereRadius ) );
}


// static
Vector2f GeometryUtils::closestPointOnSegment( const Vector2f& p, const Vector2f& v0, const Vector2f& v1 )
{
    Vector2f v01 = v1 - v0;
    float d = Vector2f::dot( v01, p - v0 );

    float t = d / v01.normSquared();
    if( t < 0.f )
    {
        t = 0.f;
    }
    else if( t > 1.f )
    {
        t = 1.f;
    }

    return v0 + t * v01;
}

// static
Vector3f GeometryUtils::closestPointOnSegment( const Vector3f& p, const Vector3f& v0, const Vector3f& v1 )
{
    Vector3f v01 = v1 - v0;
    float d = Vector3f::dot( v01, p - v0 );

    float t = d / v01.normSquared();
    if( t < 0.f )
    {
        t = 0.f;
    }
    else if( t > 1.f )
    {
        t = 1.f;
    }

    return v0 + t * v01;
}

// static
Vector2f GeometryUtils::closestPointOnTriangle( const Vector2f& p, const Vector2f& v0, const Vector2f& v1, const Vector2f& v2 )
{
    Vector2f closest0 = closestPointOnSegment( p, v0, v1 );
    Vector2f closest1 = closestPointOnSegment( p, v1, v2 );
    Vector2f closest2 = closestPointOnSegment( p, v2, v0 );

    float d0Squared = ( closest0 - p ).normSquared();
    float d1Squared = ( closest1 - p ).normSquared();
    float d2Squared = ( closest2 - p ).normSquared();

    if( d0Squared < d1Squared )
    {
        if( d0Squared < d2Squared )
        {
            return closest0;
        }
        else
        {
            return closest2;
        }
    }
    else
    {
        if( d2Squared < d1Squared )
        {
            return closest2;
        }
        else
        {
            return closest1;
        }
    }
}

// static
bool GeometryUtils::rayRayIntersection( const Vector2f& p1, const Vector2f& dir1,
                                        const Vector2f& p2, const Vector2f& dir2, Vector2f& outIntersection )
{
    Vector2f dir90 { -dir1[1], dir1[0] };
    float dirCross = Vector2f::dot( dir2, dir90 );
    if( abs( dirCross ) < EPSILON )
    {
        return false;
    }
    float param = Vector2f::dot( p1 - p2, dir90 ) / dirCross;
    if( param < 0.f )
    {
        return false;
    }

    outIntersection = p2 + param * dir2;
    return Vector2f::dot( outIntersection - p1, dir1 ) >= 0.f;
}

// static
bool GeometryUtils::lineLineSegmentIntersection( const Vector2f& p, const Vector2f& dir,
                                                 const Vector2f& p1, const Vector2f& p2, Vector2f &outIntersection)
{
    Vector2f segDir = (p2 - p1);
    float segDirLen = segDir.norm();
    if(segDirLen < EPSILON)
        return false;
    segDir = segDir / segDirLen;

    Vector2f dir90 = { -dir[1], dir[0] };
    float dirCross = Vector2f::dot(segDir, dir90);
    if(abs(dirCross) < EPSILON)
        return false;
    float param = Vector2f::dot(p - p1, dir90) / dirCross;
    if(param < 0.f || param > segDirLen)
        return false;

    outIntersection = p1 + param * segDir;

    return true;
}

// static
bool GeometryUtils::rayPlaneIntersection( const Vector3f& rayOrigin,
                                         const Vector3f& rayDirection,
                                         const Plane3f& plane,
                                         float& t )
{
    // ray: P = O + tD
    // plane: P dot N + d = 0
    // (O + tD) dot N + d = 0
    // --> t = -(O dot N + d
    float den = Vector3f::dot( rayDirection, plane.normal() );

    // TODO: epsilon
    if( abs( den ) > EPSILON )
    {
        float num = -Vector3f::dot( rayOrigin, plane.normal() ) - plane.d;
        t = num / den;
        return ( t > 0 );
    }
    return false;
}

// static
bool GeometryUtils::rayTriangleIntersection( const Vector3f& rayOrigin,
                                            const Vector3f& rayDirection,
                                            const Vector3f& v0,
                                            const Vector3f& v1,
                                            const Vector3f& v2,
                                            float& t, Vector3f& barycentrics )
{
    Vector3f edge0;
    Vector3f edge1;
    Vector3f tvec;
    Vector3f pvec;
    Vector3f qvec;

    float u;
    float v;
    float det;
    float inv_det;

    // find vectors for two edges sharing vert0
    edge0 = v1 - v0;
    edge1 = v2 - v0;

    // begin calculating determinant - also used to calculate u parameter
    pvec = Vector3f::cross( rayDirection, edge1 );
    det = Vector3f::dot( edge0, pvec );

#if 1
    // Intersect front faces only.
    if( det < EPSILON )
    {
        return false;
    }
#else
    // Allow intersection with back faces.
    if( abs( det ) < EPSILON )
    {
        return false;
    }
#endif

    // Calculate distance from vert0 to ray origin.
    tvec = rayOrigin - v0;

    // Calculate u parameter and test bounds.
    u = Vector3f::dot( tvec, pvec );
    if( u < 0.0f || u > det )
    {
        return false;
    }

    // prepare to test V parameter
    qvec = Vector3f::cross( tvec, edge0 );

    // calculate V parameter and test bounds
    v = Vector3f::dot( rayDirection, qvec );
    if( v < 0.0 || ( u + v > det ) )
    {
        return false;
    }

    // calculate t, scale parameters, ray intersects triangle
    t = Vector3f::dot( edge1, qvec );
    inv_det = 1.0f / det;
    t *= inv_det;
    u *= inv_det;
    v *= inv_det;

    barycentrics = Vector3f( 1 - u - v, u, v );
    return true;
}

// TODO: return std::pair

template< typename T >
void minMax3( const T& x0, const T& x1, const T& x2, T& minOut, T& maxOut )
{
    minOut = x0;
    maxOut = x0;

    if( x1 < minOut )
    {
        minOut = x1;
    }
    if( x1 > maxOut )
    {
        maxOut = x1;
    }
    if( x2 < minOut )
    {
        minOut = x2;
    }
    if( x2 > maxOut )
    {
        maxOut = x2;
    }
}

// TODO(jiawen): expose this
// TODO(jiawen): what is maxbox?
bool planeBoxOverlap( const Vector3f& normal, const Vector3f& vert,
    const Vector3f& maxbox )
{
    int q;
    Vector3f vmin;
    Vector3f vmax;
    float v;
    for( q = 0; q < 3; ++q )
    {
        v = vert[ q ];
        if( normal[ q ] > 0.0f )
        {
            vmin[ q ] = -maxbox[ q ] - v;
            vmax[ q ] = maxbox[ q ] - v;
        }
        else
        {
            vmin[ q ] = maxbox[ q ] - v;
            vmax[ q ] = -maxbox[ q ] - v;
        }
    }

    if( Vector3f::dot( normal, vmin ) > 0.0f )
    {
        return false;
    }

    if( Vector3f::dot( normal, vmax ) >= 0.0f )
    {
        return true;
    }

    return 0;
}

/*======================== X-tests ========================*/

#define AXISTEST_X01(a, b, fa, fb)			       \
    p0 = a*v0.y - b*v0.z;			       	   \
    p2 = a*v2.y - b*v2.z;			       	   \
    if( p0<p2 ) { min = p0; max = p2; }            \
    else { min = p2; max = p0; }                   \
    rad = fa * boxHalfSize.y + fb * boxHalfSize.z;   \
    if( min>rad || max<-rad ) return false;

#define AXISTEST_X2(a, b, fa, fb)			       \
    p0 = a*v0.y - b*v0.z;			           \
    p1 = a*v1.y - b*v1.z;			       	   \
    if( p0<p1 ) { min = p0; max = p1; }            \
    else { min = p1; max = p0; }                   \
    rad = fa * boxHalfSize.y + fb * boxHalfSize.z;   \
    if( min>rad || max<-rad ) return false;

/*======================== Y-tests ========================*/

#define AXISTEST_Y02(a, b, fa, fb)			   \
p0 = -a*v0.x + b*v0.z;		      	   \
p2 = -a*v2.x + b*v2.z;	       	       	   \
if( p0<p2 ) { min = p0; max = p2; }                    \
else { min = p2; max = p0; } \
rad = fa * boxHalfSize.x + fb * boxHalfSize.z;   \
if( min>rad || max<-rad ) return false;

#define AXISTEST_Y1(a, b, fa, fb)			   \
p0 = -a*v0.x + b*v0.z;		      	   \
p1 = -a*v1.x + b*v1.z;	     	       	   \
if( p0<p1 ) { min = p0; max = p1; } \
else { min = p1; max = p0; } \
rad = fa * boxHalfSize.x + fb * boxHalfSize.z;   \
if( min>rad || max<-rad ) return false;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)			   \
p1 = a*v1.x - b*v1.y;			           \
p2 = a*v2.x - b*v2.y;			       	   \
if( p2<p1 ) { min = p2; max = p1; }            \
else { min = p1; max = p2; } \
rad = fa * boxHalfSize.x + fb * boxHalfSize.y;   \
if( min>rad || max<-rad ) return false;

#define AXISTEST_Z0(a, b, fa, fb)			   \
p0 = a*v0.x - b*v0.y;				   \
p1 = a*v1.x - b*v1.y;			           \
if( p0<p1 ) { min = p0; max = p1; }        \
else { min = p1; max = p0; } \
rad = fa * boxHalfSize.x + fb * boxHalfSize.y;   \
if( min>rad || max<-rad ) return false;

// static
bool GeometryUtils::boxTriangleIntersection(
    const Vector3f& u0, const Vector3f& u1, const Vector3f& u2,
    const Box3f& box )
{
    // use separating axis theorem to test overlap between triangle and box
    // need to test for overlap in these directions:
    // 1) the {x,y,z}-directions (actually, since we use the AABB of the triangle
    //    we do not even need to test these)
    // 2) normal of the triangle
    // 3) crossproduct(edge from tri, {x,y,z}-direction)
    //    this gives 3x3=9 more tests

    Vector3f v0;
    Vector3f v1;
    Vector3f v2;

    float min;
    float max;
    float p0;
    float p1;
    float p2;
    float rad;

    float fex;
    float fey;
    float fez;

    Vector3f normal;
    // These are the triangle edges.
    Vector3f e0;
    Vector3f e1;
    Vector3f e2;

    Vector3f boxCenter = box.center();
    Vector3f boxHalfSize = 0.5f * box.size;

    // Move everything so that the box center is in (0,0,0).
    v0 = u0 - boxCenter;
    v1 = u0 - boxCenter;
    v2 = u0 - boxCenter;

    // Compute triangle edges.
    e0 = v1 - v0;
    e1 = v2 - v1;
    e2 = v0 - v2;

    // Bullet 3:
    // Test the 9 tests first (this was faster).
    fex = std::abs( e0.x );
    fey = std::abs( e0.y );
    fez = std::abs( e0.z );
    AXISTEST_X01( e0.z, e0.y, fez, fey );
    AXISTEST_Y02( e0.z, e0.x, fez, fex );
    AXISTEST_Z12( e0.y, e0.x, fey, fex );

    fex = std::abs( e1.x );
    fey = std::abs( e1.y );
    fez = std::abs( e1.z );
    AXISTEST_X01( e1.z, e1.y, fez, fey );
    AXISTEST_Y02( e1.z, e1.x, fez, fex );
    AXISTEST_Z0(  e1.y, e1.x, fey, fex );

    fex = std::abs( e2.x );
    fey = std::abs( e2.y );
    fez = std::abs( e2.z );
    AXISTEST_X2(  e2.z, e2.y, fez, fey );
    AXISTEST_Y1(  e2.z, e2.x, fez, fex );
    AXISTEST_Z12( e2.y, e2.x, fey, fex );


    // Bullet 1:
    //  first test overlap in the {x,y,z}-directions
    //  find min, max of the triangle each direction, and test for overlap in
    //  that direction -- this is equivalent to testing a minimal AABB around
    //  the triangle against the AABB.

    // Test in x direction.
    minMax3( v0.x, v1.x, v2.x, min, max );
    if( min > boxHalfSize.x || max < -boxHalfSize.x )
    {
        return false;
    }


    // Test in y direction.
    minMax3( v0.y, v1.y, v2.y, min, max );
    if( min > boxHalfSize.y || max < -boxHalfSize.y )
    {
        return false;
    }

    // Test in z direction.
    minMax3( v0.z, v1.z, v2.z, min, max );
    if( min > boxHalfSize.z || max < -boxHalfSize.z )
    {
        return false;
    }

    /* Bullet 2: */
    /*  test if the box intersects the plane of the triangle */
    /*  compute plane equation of triangle: normal*x+d=0 */
    normal = Vector3f::cross( e0, e1 );
    if( !planeBoxOverlap( normal, v0, boxHalfSize ) )
    {
        return false;
    }

    return true;   /* box and triangle overlaps */
}

// static
float GeometryUtils::pointToPlaneDistance( const Vector3f& point, const Vector4f& plane )
{
    // first find a point on the plane
    // we can just pick an intercept
    // for axis-aligned planes, pick another intercept until it works

    Vector3f pointOnPlane;

    if( plane[0] != 0 )
    {
        // x-intercept
        pointOnPlane = Vector3f( plane[3] / plane[0], 0, 0 );
    }
    else if( plane[1] != 0 )
    {
        pointOnPlane = Vector3f( 0, plane[3] / plane[1], 0 );
    }
    else
    {
        pointOnPlane = Vector3f( 0, 0, plane[3] / plane[1] );
    }

    Vector3f vectorToPoint = point - pointOnPlane;
    Vector3f unitNormal = plane.xyz.normalized();
    return Vector3f::dot( vectorToPoint, unitNormal );
}

// static
float GeometryUtils::pointToLineDistanceSquared( const Vector3f& point, const Vector3f& linePoint, const Vector3f& lineDir,
    Vector3f* pClosestPoint )
{
    Vector3f diff = point - linePoint;
    float dot = Vector3f::dot( diff, lineDir ); // assumes lineDir is normalized
    float distanceSquared = diff.normSquared() - dot * dot;

    if( pClosestPoint != nullptr )
    {
        *pClosestPoint = linePoint + dot * lineDir;
    }
    return distanceSquared;
}

// static
float GeometryUtils::pointToLineSegmentDistanceSquared( const Vector3f& p, const Vector3f& s0, const Vector3f& s1,
    Vector3f* pClosestPoint )
{
    Vector3f dir = s1 - s0;

    // test if vector p --> s1 points in the opposite direction of dir, if so, s1 is closer
    Vector3f dirs1 = s1 - p;
    if( Vector3f::dot( dirs1, dir ) < 0 )
    {
        if( pClosestPoint != nullptr )
        {
            *pClosestPoint = s0;
        }
        return dirs1.normSquared();
    }

    // test if vector s0 --> p points in the opposite of direction of dir, if so, s0 is closer
    Vector3f dirs0 = p - s0;
    float dot = Vector3f::dot( dirs0, dir );
    if( dot <= 0 )
    {
        if( pClosestPoint != nullptr )
        {
            *pClosestPoint = s1;
        }
        return dirs0.normSquared();
    }

    // dot = dirs0 dot dir
    //     = |dirs0| |dir| cos(t)
    // length of projection of dirs0 onto dir is |dirs0| cos(t)
    //     = dot / |dir|
    // projection itself is then
    //     = ( dot / |dir| ) * ( dir / |dir| )
    //     = dot * dir / dir.lengthSquared()
    //
    // closest approach distance is given by Pythagoras:
    // ( length of projection )^2 + ( closest approach )^2 = |dirs0|^2

    float rcpDirNormSquared = 1.0f / dir.normSquared();

    if( pClosestPoint != nullptr )
    {
        *pClosestPoint = dot * dir * rcpDirNormSquared;
    }

    // clamp to 0 just in case
    return std::max( 0.f, dirs0.normSquared() - dot * dot * rcpDirNormSquared );
}

// static
float GeometryUtils::lineToLineDistance( const Vector3f& linePoint1, const Vector3f& lineDir1, const Vector3f& linePoint2, const Vector3f& lineDir2 )
{
    //distance is along the vector perpendicular to both lines
    Vector3f dirCross = Vector3f::cross(lineDir1, lineDir2);
    float crossLength = dirCross.norm();
    if(crossLength < EPSILON) //lines are approximately parallel
        return sqrt( pointToLineDistanceSquared( linePoint1, linePoint2, lineDir2.normalized() ) );
    return abs(Vector3f::dot(linePoint2 - linePoint1, dirCross) / crossLength);
}

// static
bool GeometryUtils::raySphereIntersection( const Vector3f& rayOrigin, const Vector3f& rayDirection,
    const Vector3f& sphereCenter, float sphereRadius, float* t )
{
    Vector3f deltaOrigin = rayOrigin - sphereCenter;

    float a = rayDirection.normSquared();
    float b = 2 * Vector3f::dot( rayDirection, deltaOrigin );
    float c = deltaOrigin.normSquared() - sphereRadius * sphereRadius;

    // compute discriminant
    // if discriminant < 0, no real roots --> ray misses
    float discriminant = b * b - 4 * a * c;
    if( discriminant < 0 )
    {
        return false;
    }

    float sqrtDiscriminant = sqrt( discriminant );
    float oneOverHalfA = 0.5f / a;

    float t0 = ( -b + sqrtDiscriminant ) * oneOverHalfA;
    float t1 = ( -b - sqrtDiscriminant ) * oneOverHalfA;

    // make sure t0 < t1
    if( t0 > t1 )
    {
        swap( t0, t1 );
    }

    if( t0 > 0 )
    {
        *t = t0;
    }
    else
    {
        *t = t1;
    }

    return true;
}

// static
bool GeometryUtils::sphereSphereIntersection( const Vector3f& c0, float r0, const Vector3f& c1, float r1 )
{
    float centerDistanceSquared = ( c0 - c1 ).normSquared();
    float radiusSum = r0 + r1;
    return centerDistanceSquared <= radiusSum * radiusSum;
}

#if 0
// static
void GeometryUtils::tripleSphereIntersection( Vector3f* c0, float r0,
                                     Vector3f* c1, float r1,
                                     Vector3f* c2, float r2,
                                     int* numIntersections,
                                     Vector3f* intersect0,
                                     Vector3f* intersect1 )
{
    // compute the vector from the center of s0 --> s1
    Vector3f c0c1;
    Vector3f::subtract( c1, c0, &c0c1 );

    // distance between c0 and c1
    float d_c0c1 = c0c1.norm();
    // TODO: when does this not hold?
    float distanceToCenterOfCircle = ( d_c0c1 * d_c0c1 - r1 * r1 + r0 * r0 ) / ( 2 * d_c0c1 );

    Vector3f c0ToCenterOfCircle( &c0c1 );
    c0ToCenterOfCircle.normalize();
    c0ToCenterOfCircle.scale( distanceToCenterOfCircle );

    Vector3f centerOfCircle;
    Vector3f::add( c0, &c0ToCenterOfCircle, &centerOfCircle );

    // radius of the circle of intersection between s0 and s1
    float circleRadius = ( 1.0f / ( 2.0f * d_c0c1 ) ) * sqrt( 4 * d_c0c1 * d_c0c1 * r0 * r0 - ( d_c0c1 * d_c0c1 - r1 * r1 + r0 * r0 ) * ( d_c0c1 * d_c0c1 - r1 * r1 + r0 * r0 ) );

    // normal of plane
    Vector3f planeNormal( &c0c1 );
    planeNormal.normalize();

    // d for plane
    float planeD = Vector3f::dot( &centerOfCircle, &planeNormal );

    // compute intersection of plane with s2 to get second circle in plane
    float pseudoDistance = Vector3f::dot( &planeNormal, c2 ) - planeD;
    if( pseudoDistance < r2 )
    {
        // plane intersects sphere, center of circle is pseudodistance along -planeNormal from the center of the sphere
        Vector3f minusPseudoDistanceTimesNormal( &planeNormal );
        minusPseudoDistanceTimesNormal.scale( -pseudoDistance );

        Vector3f centerOfCircle2;
        Vector3f::add( c2, &minusPseudoDistanceTimesNormal, &centerOfCircle2 );

        float circleRadius2 = sqrt( r2 * r2 - pseudoDistance * pseudoDistance );

        Vector3f center1ToCenter2;
        Vector3f::subtract( &centerOfCircle2, &centerOfCircle, &center1ToCenter2 );
        float d_center1ToCenter2 = center1ToCenter2.norm();

        // TODO: when does this fail?
        // "d" from mathworld
        float distanceToCircleIntersection = ( d_center1ToCenter2 * d_center1ToCenter2 - circleRadius2 * circleRadius2 + circleRadius * circleRadius ) / ( 2.0f * d_center1ToCenter2 );
        Vector3f centerOfCircle1ToIntersection( &center1ToCenter2 );
        centerOfCircle1ToIntersection.normalize();
        centerOfCircle1ToIntersection.scale( distanceToCircleIntersection );

        Vector3f pointBetweenCircles;
        Vector3f::add( &centerOfCircle, &centerOfCircle1ToIntersection, &pointBetweenCircles );

        // "a / 2" from mathworld
        float distanceToActualPointFromPointBetweenCircles = ( 1.0f / ( 2.0f * d_center1ToCenter2 ) ) * sqrt( 4 * d_center1ToCenter2 * d_center1ToCenter2 * circleRadius * circleRadius - ( d_center1ToCenter2 * d_center1ToCenter2 - circleRadius2 * circleRadius2 + circleRadius * circleRadius ) * ( d_center1ToCenter2 * d_center1ToCenter2 - circleRadius2 * circleRadius2 + circleRadius * circleRadius ) );

        Vector3f vectorUpInPlane;
        Vector3f::cross( &center1ToCenter2, &planeNormal, &vectorUpInPlane );
        vectorUpInPlane.normalize();
        vectorUpInPlane.scale( distanceToActualPointFromPointBetweenCircles );

        Vector3f::add( &pointBetweenCircles, &vectorUpInPlane, intersect0 );
        vectorUpInPlane.negate();
        Vector3f::add( &pointBetweenCircles, &vectorUpInPlane, intersect1 );
    }
    else
    {
        *numIntersections = 1;
    }
}
#endif

// static
std::vector< Vector2f > GeometryUtils::uniformSampleLineSegment( const Vector2f& p0, const Vector2f& p1, int nSamples )
{
    std::vector< Vector2f > samples( nSamples );

    Vector2f unit = ( p1 - p0 ).normalized();
    float sampleSpacing = ( p1 - p0 ).norm() / ( nSamples - 1 );
    for( int i = 0; i < nSamples; ++i )
    {
        samples[ i ] = p0 + i * sampleSpacing * unit;
    }

    return samples;
}

// static
std::vector< Vector3f > GeometryUtils::uniformSampleLineSegment( const Vector3f& p0, const Vector3f& p1, int nSamples )
{
    std::vector< Vector3f > samples( nSamples );

    Vector3f unit = ( p1 - p0 ).normalized();
    float sampleSpacing = ( p1 - p0 ).norm() / ( nSamples - 1 );
    for( int i = 0; i < nSamples; ++i )
    {
        samples[ i ] = p0 + i * sampleSpacing * unit;
    }

    return samples;
}

// static
std::vector< Vector2f > GeometryUtils::uniformSampleBoxAroundLineSegment( const Vector2f& p0, const Vector2f& p1,
    float width, int nSamplesWidth, int nSamplesLength )
{
    std::vector< Vector2f > samples( nSamplesWidth * nSamplesLength );

    Vector2f unitLength = ( p1 - p0 ).normalized();
    Vector2f unitWidth = unitLength.normal().normalized();
    float sampleSpacingW = width / ( nSamplesWidth - 1 );
    float sampleSpacingL = ( p1 - p0 ).norm() / ( nSamplesLength - 1 );

    Vector2f origin = p0 - 0.5f * width * unitWidth;
    for( int w = 0; w < nSamplesWidth; ++w )
    {
        for( int l = 0; l < nSamplesLength; ++l )
        {
            samples[ w * nSamplesLength + l ] =
                origin + w * sampleSpacingW * unitWidth + l * sampleSpacingL * unitLength;
        }
    }

    return samples;
}
