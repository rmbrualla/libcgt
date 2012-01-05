#include "D3D11Mesh.h"

#include <QVector>
#include <limits>

#include <io/OBJData.h>
#include <vecmath/Matrix3f.h>

#include "D3D11Utils_Texture.h"

D3D11Mesh::D3D11Mesh( ID3D11Device* pDevice, int capacity ) :

	m_pDevice( pDevice ),
	m_vertexArray( capacity ),
	m_worldMatrix( Matrix4f::identity() )

{
	m_pDevice->AddRef();
}

// virtual
D3D11Mesh::~D3D11Mesh()
{
	m_pDevice->Release();
}

D3D11Mesh::D3D11Mesh( ID3D11Device* pDevice, VertexPosition4fNormal3fTexture2f* vertexArray, int capacity ) :

	m_pDevice( pDevice ),
	m_vertexArray( capacity ),
	m_worldMatrix( Matrix4f::identity() )

{
	m_pDevice->AddRef();

	memcpy( &( m_vertexArray[ 0 ] ), vertexArray, capacity * VertexPosition4fNormal3fTexture2f::sizeInBytes() );
	updateGPUBuffer();
	updateBoundingBox();
}

D3D11Mesh::D3D11Mesh( ID3D11Device* pDevice, std::shared_ptr< OBJData > pOBJData,
	QString texturePath, bool generatePerFaceNormalsIfNonExistent ) :

	m_pDevice( pDevice ),
	m_worldMatrix( Matrix4f::identity() )

{
	m_pDevice->AddRef();

	QVector< Vector3f >* positions = pOBJData->getPositions();
	QVector< Vector3f >* normals = pOBJData->getNormals();
	QVector< Vector2f >* texcoords = pOBJData->getTextureCoordinates();
	QHash< QString, OBJGroup* >* pGroups = pOBJData->getGroups();
	
	int nVerticesTotal = 0;
	foreach( QString groupName, pGroups->keys() )
	{
		OBJGroup* pGroup = pGroups->value( groupName );

		const auto& materials = pGroup->getMaterials();
		for( int m = 0; m < materials.size(); ++m )
		{
			QString materialName = materials[m];
			OBJMaterial* pBatchMaterial = pOBJData->getMaterial( materialName );
			auto& faces = pGroup->getFacesForMaterial( m );

			// 1 batch is 1 material for 1 group
			int nFacesInBatch = faces.size();
			int nVerticesInBatch = 3 * nFacesInBatch;

			for( int i = 0; i < nFacesInBatch; ++i )
			{
				OBJFace& face = faces[ i ];
				QVector< int >* positionIndices = face.getPositionIndices();
				QVector< int >* normalIndices = face.getNormalIndices();

				Vector3f p0 = positions->at( positionIndices->at( 0 ) );
				Vector3f p1 = positions->at( positionIndices->at( 1 ) );
				Vector3f p2 = positions->at( positionIndices->at( 2 ) );

				Vector3f n0;
				Vector3f n1;
				Vector3f n2;

				Vector2f t0;
				Vector2f t1;
				Vector2f t2;

				if( pGroup->hasNormals() )
				{
					n0 = normals->at( normalIndices->at( 0 ) );
					n1 = normals->at( normalIndices->at( 1 ) );
					n2 = normals->at( normalIndices->at( 2 ) );
				}
				else if( generatePerFaceNormalsIfNonExistent )
				{
					n0 = Vector3f::cross( p1 - p0, p2 - p0 ).normalized();
					n1 = n0;
					n2 = n0;
				}

				if( pGroup->hasTextureCoordinates() )
				{
					QVector< int >* texcoordIndices = face.getTextureCoordinateIndices();
					t0 = texcoords->at( texcoordIndices->at( 0 ) );
					t1 = texcoords->at( texcoordIndices->at( 1 ) );
					t2 = texcoords->at( texcoordIndices->at( 2 ) );
				}

				VertexPosition4fNormal3fTexture2f v0( Vector4f( p0, 1 ), n0, t0 );
				VertexPosition4fNormal3fTexture2f v1( Vector4f( p1, 1 ), n1, t1 );
				VertexPosition4fNormal3fTexture2f v2( Vector4f( p2, 1 ), n2, t2 );

				m_vertexArray.push_back( v0 );
				m_vertexArray.push_back( v1 );
				m_vertexArray.push_back( v2 );
			}

			QString diffuseTextureFilename = pBatchMaterial->diffuseTexture();
			std::shared_ptr< DynamicTexture2D > diffuseTexture;
			if( !( diffuseTextureFilename.isNull() ) )
			{
				if( !( m_diffuseTextures.contains( diffuseTextureFilename ) ) )
				{
					QString absFilename = texturePath + diffuseTextureFilename;
					m_diffuseTextures.insert( diffuseTextureFilename, D3D11Utils_Texture::createTextureFromFile( m_pDevice, absFilename ) );
				}
				diffuseTexture = m_diffuseTextures[ diffuseTextureFilename ];
			}

			Vector3f kd = pBatchMaterial->diffuseColor();
			//printf( "Group %s, material %s, using kd = %s\n",
			//	qPrintable( groupName ), qPrintable( materialName ), qPrintable( kd.toString() ) );			 

			addVertexRange( Vector2i( nVerticesTotal, nVerticesInBatch ),
				kd, diffuseTexture );
		}
	}

	updateGPUBuffer();
	updateBoundingBox();	
}

int D3D11Mesh::capacity() const
{
	return m_vertexArray.size();
}

std::vector< Vector2i >& D3D11Mesh::vertexRanges()
{
	return m_vertexRanges;
}

std::vector< Vector3f >& D3D11Mesh::diffuseColors()
{
	return m_diffuseColors;
}

void D3D11Mesh::addVertexRange( const Vector2i& vr,
	const Vector3f& kd,
	std::shared_ptr< DynamicTexture2D > pDiffuseTexture )
{
	m_vertexRanges.push_back( vr );
	m_diffuseColors.push_back( kd );
	m_diffuseTexturesRanges.push_back( pDiffuseTexture );
}

std::vector< VertexPosition4fNormal3fTexture2f >& D3D11Mesh::vertexArray()
{
	return m_vertexArray;
}

const std::vector< VertexPosition4fNormal3fTexture2f >& D3D11Mesh::vertexArray() const
{
	return m_vertexArray;
}

Matrix4f D3D11Mesh::twoUnitCubeWorldMatrix() const
{
	Matrix4f tr = Matrix4f::translation( -m_boundingBox.center() );
	Matrix4f s = Matrix4f::uniformScaling( 2.0f / m_boundingBox.longestSideLength() );
	return s * tr;
}

const BoundingBox3f& D3D11Mesh::boundingBox() const
{
	return m_boundingBox;
}

Matrix4f D3D11Mesh::worldMatrix() const
{
	return m_worldMatrix;
}

void D3D11Mesh::setWorldMatrix( const Matrix4f& m )
{
	m_worldMatrix = m;
}

Matrix3f D3D11Mesh::normalMatrix() const
{
	return worldMatrix().getSubmatrix3x3( 0, 0 ).inverse().transposed();
}

Matrix4f D3D11Mesh::normalMatrix4x4() const
{
	Matrix4f n;
	n.setSubmatrix3x3( 0, 0, normalMatrix() );
	return n;
}

std::shared_ptr< DynamicVertexBuffer > D3D11Mesh::vertexBuffer() const
{
	return m_pVertexBuffer;
}

void D3D11Mesh::updateGPUBuffer()
{
	if( m_pVertexBuffer.get() == nullptr ||
		m_pVertexBuffer->capacity() < m_vertexArray.size() )
	{
		m_pVertexBuffer.reset( new DynamicVertexBuffer( m_pDevice, static_cast< int >( m_vertexArray.size() ), VertexPosition4fNormal3fTexture2f::sizeInBytes() ) );
	}

	VertexPosition4fNormal3fTexture2f* gpuVertexArray = reinterpret_cast< VertexPosition4fNormal3fTexture2f* >( m_pVertexBuffer->mapForWriteDiscard().pData );
	memcpy( gpuVertexArray, &( m_vertexArray[ 0 ] ), m_vertexArray.size() * VertexPosition4fNormal3fTexture2f::sizeInBytes() );
	m_pVertexBuffer->unmap();
}

void D3D11Mesh::updateBoundingBox()
{
	// compute bounding box	
	float xMin = ( std::numeric_limits< float >::max )();
	float xMax = std::numeric_limits< float >::lowest();
	float yMin = ( std::numeric_limits< float >::max )();
	float yMax = std::numeric_limits< float >::lowest();
	float zMin = ( std::numeric_limits< float >::max )();
	float zMax = std::numeric_limits< float >::lowest();

	for( int i = 0; i < m_vertexArray.size(); ++i )
	{
		Vector4f xyzw = m_vertexArray[ i ].position;
		float x = xyzw.x;
		float y = xyzw.y;
		float z = xyzw.z;

		if( x < xMin )
		{
			xMin = x;
		}
		if( x > xMax )
		{
			xMax = x;
		}
		if( y < yMin )
		{
			yMin = y;
		}
		if( y > yMax )
		{
			yMax = y;
		}
		if( z < zMin )
		{
			zMin = z;
		}
		if( z > zMax )
		{
			zMax = z;
		}
	}

	m_boundingBox = BoundingBox3f( xMin, yMin, zMin, xMax, yMax, zMax );
}
