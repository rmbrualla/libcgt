#pragma once

class Vector2i;
class Vector3d;

class Vector2d
{
public:

    Vector2d();
    Vector2d( double x, double y );
    Vector2d( const Vector2d& rv ); // copy constructor
    Vector2d& operator = ( const Vector2d& rv ); // assignment operator
    // no destructor necessary

    // returns the ith element
    const double& operator [] ( int i ) const;
    double& operator [] ( int i );

    Vector2d xy() const;
    Vector2d yx() const;
    Vector2d xx() const;
    Vector2d yy() const;

    double norm() const;
    double normSquared() const;
    void normalize();
    Vector2d normalized() const;

    // ---- Utility ----
    operator const double* (); // automatic type conversion for GL
    void print() const;

    static double dot( const Vector2d& v0, const Vector2d& v1 );

    static Vector3d cross( const Vector2d& v0, const Vector2d& v1 );

    union
    {
        struct
        {
            double x;
            double y;
        };
        double m_elements[ 2 ];
    };

};

Vector2d operator + ( const Vector2d& v0, const Vector2d& v1 );
Vector2d operator - ( const Vector2d& v0, const Vector2d& v1 );
Vector2d operator * ( const Vector2d& v0, const Vector2d& v1 );
Vector2d operator / ( const Vector2d& v0, const Vector2d& v1 );

Vector2d operator - ( const Vector2d& v );
Vector2d operator * ( double d, const Vector2d& v );
Vector2d operator * ( const Vector2d& v, double d );
