#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>
#include "GLImageInternalFormat.h"
#include "GLImageFormat.h"

class GLTexture
{
public:

    enum class Target
    {
        // Standard targets.
        TEXTURE_1D = GL_TEXTURE_1D,
        TEXTURE_2D = GL_TEXTURE_2D,
        TEXTURE_3D = GL_TEXTURE_3D,

        // Special geometric targets.
        TEXTURE_CUBE_MAP = GL_TEXTURE_CUBE_MAP,
        TEXTURE_RECTANGLE = GL_TEXTURE_RECTANGLE, // basically deprecated

        // Array targets.
        TEXTURE_1D_ARRAY = GL_TEXTURE_1D_ARRAY,
        TEXTURE_2D_ARRAY = GL_TEXTURE_2D_ARRAY,
        TEXTURE_CUBE_MAP_ARRAY = GL_TEXTURE_CUBE_MAP_ARRAY,

        // Multisample targets.
        TEXTURE_2D_MULTISAMPLE = GL_TEXTURE_2D_MULTISAMPLE,
        TEXTURE_2D_MULTISAMPLE_ARRAY = GL_TEXTURE_2D_MULTISAMPLE_ARRAY,

        // Texture buffer.
        TEXTURE_BUFFER = GL_TEXTURE_BUFFER
    };

	// Returns the current active texture unit.
	static GLenum activeTextureUnit();

	// Returns the maximum number of texture image units
	// that can be bound per pipeline stage.
	static int maxTextureImageUnits();

	// Returns the maximum number of texture image units
	// across the entire pipeline.
	static int maxCombinedTextureImageUnits();
	
    // TODO: make it 1d and 2d, returning the same thing.
	// Max width and height
	static int maxSize1D2D();

	// Max width, height, and depth
	static int maxSize3D();

	// Max for any face
	static int maxSizeCubeMap();

	virtual ~GLTexture();

	// Binds this texture object to the texture unit;
	void bind( GLenum texunit = GL_TEXTURE0 );

	// Unbinds this texture from the texture unit.
	void unbind( GLenum texunit = GL_TEXTURE0 );

	GLuint id();
	GLenum target(); // TODO: make target also enum class
	GLImageInternalFormat internalFormat();

protected:
	
	GLTexture( GLenum target, GLImageInternalFormat internalFormat );

private:

	GLenum m_target;
	GLuint m_id;	
	GLImageInternalFormat m_internalFormat;
};
