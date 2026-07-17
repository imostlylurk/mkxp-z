/*
** shader.cpp
**
** This file is part of mkxp.
**
** Copyright (C) 2013 - 2021 Amaryllis Kulla <ancurio@mapleshrine.eu>
**
** mkxp is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 2 of the License, or
** (at your option) any later version.
**
** mkxp is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with mkxp.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "shader.h"
#include "config.h"
#include "graphics.h"
#include "sharedstate.h"
#include "glstate.h"
#include "exception.h"

#include <assert.h>
#include <string.h>
#include <iostream>

#include "common.h.xxd"
#include "sprite.frag.xxd"
#include "hue.frag.xxd"
#include "trans.frag.xxd"
#include "transSimple.frag.xxd"
#include "bitmapBlit.frag.xxd"
#include "plane.frag.xxd"
#include "gray.frag.xxd"
#include "flatColor.frag.xxd"
#include "simple.frag.xxd"
#include "simpleColor.frag.xxd"
#include "simpleAlpha.frag.xxd"
#include "simpleAlphaUni.frag.xxd"
#include "tilemap.frag.xxd"
#include "flashMap.frag.xxd"
#include "bicubic.frag.xxd"
#include "lanczos3.frag.xxd"
#ifdef MKXPZ_HAVE_EXTRA_SHADERS
#include "xbrz.frag.xxd"
#include "nis.frag.xxd"
#endif
#include "minimal.vert.xxd"
#include "simple.vert.xxd"
#include "simpleColor.vert.xxd"
#include "sprite.vert.xxd"
#include "tilemap.vert.xxd"
#include "blur.frag.xxd"
#include "simpleMatrix.vert.xxd"
#include "blurH.vert.xxd"
#include "blurV.vert.xxd"
#include "tilemapvx.vert.xxd"
#include "kglInvert.frag.xxd"
#include "kglCompressAlpha.frag.xxd"
#include "kglSubtract.frag.xxd"
#include "kglShadowH.frag.xxd"
#include "kglShadowV.frag.xxd"

#define INIT_SHADER(vert, frag, name) \
{ \
	Shader::init(mkxp_shader_##vert##_vert, mkxp_shader_##vert##_vert_len, mkxp_shader_##frag##_frag, mkxp_shader_##frag##_frag_len, \
	#vert, #frag, #name); \
}

#define GET_U(name) u_##name = gl.GetUniformLocation(program, #name)

static void printShaderLog(GLuint shader)
{
	GLint logLength;
	gl.GetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

	std::string log(logLength, '\0');
	gl.GetShaderInfoLog(shader, log.size(), 0, &log[0]);

	std::clog << "Shader log:\n" << log;
}

static void printProgramLog(GLuint program)
{
	GLint logLength;
	gl.GetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

	std::string log(logLength, '\0');
	gl.GetProgramInfoLog(program, log.size(), 0, &log[0]);

	std::clog << "Program log:\n" << log;
}

Shader::Shader() : initialized(false)
{
	vertShader = gl.CreateShader(GL_VERTEX_SHADER);
	fragShader = gl.CreateShader(GL_FRAGMENT_SHADER);

	program = gl.CreateProgram();
}

Shader::~Shader()
{
	gl.DeleteProgram(program);
	gl.DeleteShader(vertShader);
	gl.DeleteShader(fragShader);
}

void Shader::bind()
{
	glState.program.set(program);
}

void Shader::unbind()
{
	gl.ActiveTexture(GL_TEXTURE0);
	glState.program.set(0);
}

static void setupShaderSource(GLuint shader, GLenum type,
                              const unsigned char *body, int bodySize)
{
	static const char glesDefine[] = "#define GLSLES\n";
	static const char fragDefine[] = "#define FRAGMENT_SHADER\n";
	static const char version130[] = "#version 130\n";

	const GLchar *shaderSrc[5];
	GLint shaderSrcSize[5];
	size_t i = 0;

	if (!gl.glsles)
	{
		/* Desktop GL: request GLSL 1.30 for dynamic array indexing support.
		 * GLSL 1.30 is backwards-compatible with 1.10 syntax (texture2D,
		 * gl_FragColor, varying) while allowing variable array subscripts. */
		shaderSrc[i] = version130;
		shaderSrcSize[i] = sizeof(version130)-1;
		++i;
	}

	if (gl.glsles)
	{
		shaderSrc[i] = glesDefine;
		shaderSrcSize[i] = sizeof(glesDefine)-1;
		++i;
	}

	if (type == GL_FRAGMENT_SHADER)
	{
		shaderSrc[i] = fragDefine;
		shaderSrcSize[i] = sizeof(fragDefine)-1;
		++i;
	}

	shaderSrc[i] = (const GLchar*) mkxp_shader_common_h;
	shaderSrcSize[i] = mkxp_shader_common_h_len;
	++i;

	shaderSrc[i] = (const GLchar*) body;
	shaderSrcSize[i] = bodySize;
	++i;

	gl.ShaderSource(shader, i, shaderSrc, shaderSrcSize);
}

void Shader::init(const unsigned char *vert, int vertSize,
                  const unsigned char *frag, int fragSize,
                  const char *vertName, const char *fragName,
                  const char *programName)
{
	if (initialized)
	{
		/* Calling Shader::init() more than once causes a small number of graphics drivers to encounter linking errors.
		 * In particular, the Nintendo Switch homebrew toolchain's Mesa driver has this problem.
		 * So we throw this exception on every platform to reduce the probability of regressions. */
		throw Exception(Exception::MKXPError,
	                    "Attempted to call Shader::init() more than once");
	}

	GLint success;

	/* Compile vertex shader */
	setupShaderSource(vertShader, GL_VERTEX_SHADER, vert, vertSize);
	gl.CompileShader(vertShader);

	gl.GetShaderiv(vertShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		printShaderLog(vertShader);
		throw Exception(Exception::MKXPError,
	                    "GLSL: An error occurred while compiling vertex shader '%s' in program '%s'",
	                    vertName, programName);
	}

	/* Compile fragment shader */
	setupShaderSource(fragShader, GL_FRAGMENT_SHADER, frag, fragSize);
	gl.CompileShader(fragShader);

	gl.GetShaderiv(fragShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		printShaderLog(fragShader);
		throw Exception(Exception::MKXPError,
	                    "GLSL: An error occurred while compiling fragment shader '%s' in program '%s'",
	                    fragName, programName);
	}

	/* Link shader program */
	gl.AttachShader(program, vertShader);
	gl.AttachShader(program, fragShader);

	gl.BindAttribLocation(program, Position, "position");
	gl.BindAttribLocation(program, TexCoord, "texCoord");
	gl.BindAttribLocation(program, Color, "color");

	gl.LinkProgram(program);

	gl.GetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success)
	{
		printProgramLog(program);
		throw Exception(Exception::MKXPError,
	                    "GLSL: An error occurred while linking program '%s' (vertex '%s', fragment '%s')",
	                    programName, vertName, fragName);
	}

	initialized = true;
}

void Shader::initFromFile(const char *_vertFile, const char *_fragFile,
                          const char *programName)
{
	std::string vertContents, fragContents;
	readFile(_vertFile, vertContents);
	readFile(_fragFile, fragContents);

	init((const unsigned char*) vertContents.c_str(), vertContents.size(),
	     (const unsigned char*) fragContents.c_str(), fragContents.size(),
	     _vertFile, _fragFile, programName);
}

void Shader::setVec2Uniform(GLint location, const Vec2 &vec)
{
    gl.Uniform2f(location, vec.x, vec.y);
}

void Shader::setVec4Uniform(GLint location, const Vec4 &vec)
{
	gl.Uniform4f(location, vec.x, vec.y, vec.z, vec.w);
}

void Shader::setTexUniform(GLint location, unsigned unitIndex, TEX::ID texture)
{
	GLenum texUnit = GL_TEXTURE0 + unitIndex;

	gl.ActiveTexture(texUnit);
	gl.BindTexture(GL_TEXTURE_2D, texture.gl);
	gl.Uniform1i(location, unitIndex);
	gl.ActiveTexture(GL_TEXTURE0);
}

void ShaderBase::GLProjMat::apply(const Vec2i &value)
{
	/* glOrtho replacement */
	const float a = 2.f / value.x;
	const float b = 2.f / value.y;
	const float c = -2.f;

	GLfloat mat[16] =
	{
		 a,  0,  0,  0,
		 0,  b,  0,  0,
		 0,  0,  c,  0,
		-1, -1, -1,  1
	};

	gl.UniformMatrix4fv(u_mat, 1, GL_FALSE, mat);
}

void ShaderBase::init()
{
	GET_U(texSizeInv);
	GET_U(translation);

	projMat.u_mat = gl.GetUniformLocation(program, "projMat");
}

void ShaderBase::applyViewportProj()
{
	// High-res: scale the matrix if we're rendering to the PingPong framebuffer.
	const IntRect &vp = glState.viewport.get();
	if (shState->config().enableHires && shState->graphics().isPingPongFramebufferActive() && framebufferScalingAllowed()) {
		projMat.set(Vec2i(shState->graphics().width(), shState->graphics().height()));
	}
	else {
		projMat.set(Vec2i(vp.w, vp.h));
	}
}

bool ShaderBase::framebufferScalingAllowed()
{
	return true;
}

void ShaderBase::setTexSize(const Vec2i &value)
{
	gl.Uniform2f(u_texSizeInv, 1.f / value.x, 1.f / value.y);
}

void ShaderBase::setTranslation(const Vec2i &value)
{
	gl.Uniform2f(u_translation, value.x, value.y);
}


FlatColorShader::FlatColorShader()
{
	INIT_SHADER(minimal, flatColor, FlatColorShader);

	ShaderBase::init();

	GET_U(color);
}

void FlatColorShader::setColor(const Vec4 &value)
{
	setVec4Uniform(u_color, value);
}


SimpleShader::SimpleShader()
{
	INIT_SHADER(simple, simple, SimpleShader);

	ShaderBase::init();

	GET_U(texOffsetX);
}

SimpleShader::SimpleShader(const ShaderNoConstructTag &)
{
}

void SimpleShader::setTexOffsetX(int value)
{
	gl.Uniform1f(u_texOffsetX, value);
}


SimpleColorShader::SimpleColorShader()
{
	INIT_SHADER(simpleColor, simpleColor, SimpleColorShader);

	ShaderBase::init();
}


SimpleAlphaShader::SimpleAlphaShader()
{
	INIT_SHADER(simpleColor, simpleAlpha, SimpleAlphaShader);

	ShaderBase::init();
}


SimpleSpriteShader::SimpleSpriteShader()
{
	INIT_SHADER(sprite, simple, SimpleSpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
}

SimpleSpriteShader::SimpleSpriteShader(const ShaderNoConstructTag &)
{
}

void SimpleSpriteShader::setSpriteMat(const float value[16])
{
	gl.UniformMatrix4fv(u_spriteMat, 1, GL_FALSE, value);
}

BicubicSpriteShader::BicubicSpriteShader() : Lanczos3SpriteShader(ShaderNoConstructTag())
{
	INIT_SHADER(sprite, bicubic, BicubicSpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
	GET_U(sourceSize);
	GET_U(bc);
}

void BicubicSpriteShader::setSharpness(int sharpness)
{
	gl.Uniform2f(u_bc, 1.f - sharpness * 0.01f, sharpness * 0.005f);
}

Lanczos3SpriteShader::Lanczos3SpriteShader() : SimpleSpriteShader(ShaderNoConstructTag())
{
	INIT_SHADER(sprite, lanczos3, Lanczos3SpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
	GET_U(sourceSize);
}

Lanczos3SpriteShader::Lanczos3SpriteShader(const ShaderNoConstructTag &) : SimpleSpriteShader(ShaderNoConstructTag())
{
}

void Lanczos3SpriteShader::setTexSize(const Vec2i &value)
{
	ShaderBase::setTexSize(value);
	gl.Uniform2f(u_sourceSize, (float)value.x, (float)value.y);
}

#ifdef MKXPZ_HAVE_EXTRA_SHADERS
XbrzSpriteShader::XbrzSpriteShader() : Lanczos3SpriteShader(ShaderNoConstructTag())
{
	INIT_SHADER(sprite, xbrz, XbrzSpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
	GET_U(sourceSize);
	GET_U(targetScale);
}

void XbrzSpriteShader::setTargetScale(const Vec2 &value)
{
	gl.Uniform2f(u_targetScale, value.x, value.y);
}

NisSpriteShader::NisSpriteShader() : Lanczos3SpriteShader(ShaderNoConstructTag())
{
	INIT_SHADER(sprite, nis, NisSpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
	GET_U(sourceSize);
	GET_U(targetScale);
	GET_U(sharpness);
}

void NisSpriteShader::setTargetScale(const Vec2 &value)
{
	gl.Uniform2f(u_targetScale, value.x, value.y);
}

void NisSpriteShader::setSharpness(float value)
{
	gl.Uniform1f(u_sharpness, value);
}
#endif

AlphaSpriteShader::AlphaSpriteShader()
{
	INIT_SHADER(sprite, simpleAlphaUni, AlphaSpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
	GET_U(alpha);
}

void AlphaSpriteShader::setSpriteMat(const float value[16])
{
	gl.UniformMatrix4fv(u_spriteMat, 1, GL_FALSE, value);
}

void AlphaSpriteShader::setAlpha(float value)
{
	gl.Uniform1f(u_alpha, value);
}


TransShader::TransShader()
{
	INIT_SHADER(simple, trans, TransShader);

	ShaderBase::init();

	GET_U(currentScene);
	GET_U(frozenScene);
	GET_U(transMap);
	GET_U(prog);
	GET_U(vague);
}

void TransShader::setCurrentScene(TEX::ID tex)
{
	setTexUniform(u_currentScene, 1, tex);
}

void TransShader::setFrozenScene(TEX::ID tex)
{
	setTexUniform(u_frozenScene, 2, tex);
}

void TransShader::setTransMap(TEX::ID tex)
{
	setTexUniform(u_transMap, 3, tex);
}

void TransShader::setProg(float value)
{
	gl.Uniform1f(u_prog, value);
}

void TransShader::setVague(float value)
{
	gl.Uniform1f(u_vague, value);
}


SimpleTransShader::SimpleTransShader()
{
	INIT_SHADER(simple, transSimple, SimpleTransShader);

	ShaderBase::init();

	GET_U(currentScene);
	GET_U(frozenScene);
	GET_U(prog);
}

void SimpleTransShader::setCurrentScene(TEX::ID tex)
{
	setTexUniform(u_currentScene, 1, tex);
}

void SimpleTransShader::setFrozenScene(TEX::ID tex)
{
	setTexUniform(u_frozenScene, 2, tex);
}

void SimpleTransShader::setProg(float value)
{
	gl.Uniform1f(u_prog, value);
}


SpriteShader::SpriteShader()
{
	INIT_SHADER(sprite, sprite, SpriteShader);

	ShaderBase::init();

	GET_U(spriteMat);
	GET_U(tone);
	GET_U(color);
	GET_U(opacity);
	GET_U(bushY);
	GET_U(bushUnder);
	GET_U(bushSlope);
	GET_U(bushIntercept);
	GET_U(bushOpacity);
    GET_U(pattern);
    GET_U(patternBlendType);
    GET_U(patternTile);
    GET_U(renderPattern);
    GET_U(patternSizeInv);
    GET_U(patternOpacity);
    GET_U(patternScroll);
    GET_U(patternZoom);
    GET_U(invert);
}

void SpriteShader::setSpriteMat(const float value[16])
{
	gl.UniformMatrix4fv(u_spriteMat, 1, GL_FALSE, value);
}

void SpriteShader::setTone(const Vec4 &tone)
{
	setVec4Uniform(u_tone, tone);
}

void SpriteShader::setColor(const Vec4 &color)
{
	setVec4Uniform(u_color, color);
}

void SpriteShader::setOpacity(float value)
{
	gl.Uniform1f(u_opacity, value);
}

void SpriteShader::setBushDepth(bool bushY, bool bushUnder, float bushSlope, float bushIntercept)
{
	gl.Uniform1f(u_bushY, bushY);
	gl.Uniform1f(u_bushUnder, bushUnder);
	gl.Uniform1f(u_bushSlope, bushSlope);
	gl.Uniform1f(u_bushIntercept, bushIntercept);
}

void SpriteShader::setBushOpacity(float value)
{
	gl.Uniform1f(u_bushOpacity, value);
}

void SpriteShader::setPattern(const TEX::ID pattern, const Vec2 &dimensions)
{
    setTexUniform(u_pattern, 1, pattern);
    gl.Uniform2f(u_patternSizeInv, 1.f / dimensions.x, 1.f / dimensions.y);
}

void SpriteShader::setPatternBlendType(int blendType)
{
    gl.Uniform1i(u_patternBlendType, blendType);
}

void SpriteShader::setPatternTile(bool value)
{
    gl.Uniform1i(u_patternTile, value);
}

void SpriteShader::setShouldRenderPattern(bool value)
{
    gl.Uniform1i(u_renderPattern, value);
}

void SpriteShader::setPatternOpacity(float value)
{
    gl.Uniform1f(u_patternOpacity, value);
}

void SpriteShader::setPatternScroll(const Vec2 &scroll)
{
    setVec2Uniform(u_patternScroll, scroll);
}

void SpriteShader::setPatternZoom(const Vec2 &zoom)
{
    setVec2Uniform(u_patternZoom, zoom);
}

void SpriteShader::setInvert(bool value)
{
    gl.Uniform1i(u_invert, value);
}


PlaneShader::PlaneShader()
{
	INIT_SHADER(simple, plane, PlaneShader);

	ShaderBase::init();

	GET_U(tone);
	GET_U(color);
	GET_U(flash);
	GET_U(opacity);
}

void PlaneShader::setTone(const Vec4 &tone)
{
	setVec4Uniform(u_tone, tone);
}

void PlaneShader::setColor(const Vec4 &color)
{
	setVec4Uniform(u_color, color);
}

void PlaneShader::setFlash(const Vec4 &flash)
{
	setVec4Uniform(u_flash, flash);
}

void PlaneShader::setOpacity(float value)
{
	gl.Uniform1f(u_opacity, value);
}


GrayShader::GrayShader()
{
	INIT_SHADER(simple, gray, GrayShader);

	ShaderBase::init();

	GET_U(gray);
}

bool GrayShader::framebufferScalingAllowed()
{
	// This shader is used with input textures that have already had a
	// framebuffer scale applied. So we don't want to double-apply it.
	return false;
}

void GrayShader::setGray(float value)
{
	gl.Uniform1f(u_gray, value);
}


TilemapShader::TilemapShader()
{
	INIT_SHADER(tilemap, tilemap, TilemapShader);

	ShaderBase::init();

	GET_U(tone);
	GET_U(color);
	GET_U(opacity);

	GET_U(aniIndex);
	GET_U(atFrames);
}

void TilemapShader::setTone(const Vec4 &tone)
{
	setVec4Uniform(u_tone, tone);
}

void TilemapShader::setColor(const Vec4 &color)
{
	setVec4Uniform(u_color, color);
}

void TilemapShader::setOpacity(float value)
{
	gl.Uniform1f(u_opacity, value);
}

void TilemapShader::setAniIndex(int value)
{
	gl.Uniform1i(u_aniIndex, value);
}

void TilemapShader::setATFrames(int values[7])
{
	gl.Uniform1iv(u_atFrames, 7, values);
}



FlashMapShader::FlashMapShader()
{
	INIT_SHADER(simpleColor, flashMap, FlashMapShader);

	ShaderBase::init();

	GET_U(alpha);
}

void FlashMapShader::setAlpha(float value)
{
	gl.Uniform1f(u_alpha, value);
}


HueShader::HueShader()
{
	INIT_SHADER(simple, hue, HueShader);

	ShaderBase::init();

	GET_U(hueAdjust);
}

void HueShader::setHueAdjust(float value)
{
	gl.Uniform1f(u_hueAdjust, value);
}


SimpleMatrixShader::SimpleMatrixShader()
{
	INIT_SHADER(simpleMatrix, simpleAlpha, SimpleMatrixShader);

	ShaderBase::init();

	GET_U(matrix);
}

void SimpleMatrixShader::setMatrix(const float value[16])
{
	gl.UniformMatrix4fv(u_matrix, 1, GL_FALSE, value);
}


BlurShader::HPass::HPass()
{
	INIT_SHADER(blurH, blur, BlurShader::HPass);

	ShaderBase::init();
}

BlurShader::VPass::VPass()
{
	INIT_SHADER(blurV, blur, BlurShader::VPass);

	ShaderBase::init();
}


TilemapVXShader::TilemapVXShader()
{
	INIT_SHADER(tilemapvx, simple, TilemapVXShader);

	ShaderBase::init();

	GET_U(aniOffset);
}

void TilemapVXShader::setAniOffset(const Vec2 &value)
{
	gl.Uniform2f(u_aniOffset, value.x, value.y);
}


BltShader::BltShader()
{
	INIT_SHADER(simple, bitmapBlit, BltShader);

	init();
}

BltShader::BltShader(const ShaderNoConstructTag &)
{
}

void BltShader::init()
{
	ShaderBase::init();

	GET_U(source);
	GET_U(destination);
	GET_U(subRect);
	GET_U(opacity);
}

void BltShader::setSource()
{
	gl.Uniform1i(u_source, 0);
}

void BltShader::setDestination(const TEX::ID value)
{
	setTexUniform(u_destination, 1, value);
}

void BltShader::setSubRect(const FloatRect &value)
{
	gl.Uniform4f(u_subRect, value.x, value.y, value.w, value.h);
}

void BltShader::setOpacity(float value)
{
	gl.Uniform1f(u_opacity, value);
}

KglInvertShader::KglInvertShader()
{
	INIT_SHADER(simple, kglInvert, KglInvertShader);

	ShaderBase::init();
}

KglCompressAlphaShader::KglCompressAlphaShader()
{
	INIT_SHADER(simple, kglCompressAlpha, KglCompressAlphaShader);

	ShaderBase::init();
}

KglSubtractShader::KglSubtractShader() : BltShader(ShaderNoConstructTag())
{
	INIT_SHADER(simple, kglSubtract, KglSubtractShader);

	BltShader::init();
}

KglShadowShaderH::KglShadowShaderH()
{
	INIT_SHADER(simple, kglShadowH, KglShadowShaderH);

	ShaderBase::init();

	GET_U(x1);
	GET_U(x2);
	GET_U(y);
	GET_U(soft);
	GET_U(w);
	GET_U(h);
	GET_U(x_center);
	GET_U(y_center);
	GET_U(slope1);
	GET_U(slope2);
}

void KglShadowShaderH::setParams(int x1, int x2, int y, bool soft, int w, int h, int x_center, int y_center, double slope1, double slope2)
{
	gl.Uniform1i(u_x1, x1);
	gl.Uniform1i(u_x2, x2);
	gl.Uniform1i(u_y, y);
	gl.Uniform1i(u_soft, soft);
	gl.Uniform1i(u_w, w);
	gl.Uniform1i(u_h, h);
	gl.Uniform1i(u_x_center, x_center);
	gl.Uniform1i(u_y_center, y_center);
	gl.Uniform1f(u_slope1, slope1);
	gl.Uniform1f(u_slope2, slope2);
}

KglShadowShaderV::KglShadowShaderV()
{
	INIT_SHADER(simple, kglShadowV, KglShadowShaderV);

	ShaderBase::init();

	GET_U(y1);
	GET_U(y2);
	GET_U(x);
	GET_U(wall);
	GET_U(soft);
	GET_U(w);
	GET_U(h);
	GET_U(x_center);
	GET_U(y_center);
	GET_U(slope1);
	GET_U(slope2);
}

void KglShadowShaderV::setParams(int y1, int y2, int x, bool wall, bool soft, int w, int h, int x_center, int y_center, double slope1, double slope2)
{
	gl.Uniform1i(u_y1, y1);
	gl.Uniform1i(u_y2, y2);
	gl.Uniform1i(u_x, x);
	gl.Uniform1i(u_wall, wall);
	gl.Uniform1i(u_soft, soft);
	gl.Uniform1i(u_w, w);
	gl.Uniform1i(u_h, h);
	gl.Uniform1i(u_x_center, x_center);
	gl.Uniform1i(u_y_center, y_center);
	gl.Uniform1f(u_slope1, slope1);
	gl.Uniform1f(u_slope2, slope2);
}

BicubicShader::BicubicShader() : Lanczos3Shader(ShaderNoConstructTag())
{
	INIT_SHADER(simple, bicubic, BicubicShader);

	ShaderBase::init();

	GET_U(texOffsetX);
	GET_U(sourceSize);
	GET_U(bc);
}

void BicubicShader::setSharpness(int sharpness)
{
	gl.Uniform2f(u_bc, 1.f - sharpness * 0.01f, sharpness * 0.005f);
}

Lanczos3Shader::Lanczos3Shader() : SimpleShader(ShaderNoConstructTag())
{
	INIT_SHADER(simple, lanczos3, Lanczos3Shader);

	ShaderBase::init();

	GET_U(texOffsetX);
	GET_U(sourceSize);
}

Lanczos3Shader::Lanczos3Shader(const ShaderNoConstructTag &) : SimpleShader(ShaderNoConstructTag())
{
}

void Lanczos3Shader::setTexSize(const Vec2i &value)
{
	ShaderBase::setTexSize(value);
	gl.Uniform2f(u_sourceSize, (float)value.x, (float)value.y);
}

#ifdef MKXPZ_HAVE_EXTRA_SHADERS
XbrzShader::XbrzShader() : Lanczos3Shader(ShaderNoConstructTag())
{
	INIT_SHADER(simple, xbrz, XbrzShader);

	ShaderBase::init();

	GET_U(texOffsetX);
	GET_U(sourceSize);
	GET_U(targetScale);
}

void XbrzShader::setTargetScale(const Vec2 &value)
{
	gl.Uniform2f(u_targetScale, value.x, value.y);
}

NisShader::NisShader() : Lanczos3Shader(ShaderNoConstructTag())
{
	INIT_SHADER(simple, nis, NisShader);

	ShaderBase::init();

	GET_U(texOffsetX);
	GET_U(sourceSize);
	GET_U(targetScale);
	GET_U(sharpness);

	// Upload NIS coefficient tables
    static const float coef_scale_data[384] = {
        0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.0000f, 0.0000f,
        0.0029f, -0.0127f, 1.0000f, 0.0132f, -0.0034f, 0.0000f,
        0.0063f, -0.0249f, 0.9985f, 0.0269f, -0.0068f, 0.0000f,
        0.0088f, -0.0361f, 0.9956f, 0.0415f, -0.0103f, 0.0005f,
        0.0117f, -0.0474f, 0.9932f, 0.0562f, -0.0142f, 0.0005f,
        0.0142f, -0.0576f, 0.9897f, 0.0713f, -0.0181f, 0.0005f,
        0.0166f, -0.0674f, 0.9844f, 0.0874f, -0.0220f, 0.0010f,
        0.0186f, -0.0762f, 0.9785f, 0.1040f, -0.0264f, 0.0015f,
        0.0205f, -0.0850f, 0.9727f, 0.1206f, -0.0308f, 0.0020f,
        0.0225f, -0.0928f, 0.9648f, 0.1382f, -0.0352f, 0.0024f,
        0.0239f, -0.1006f, 0.9575f, 0.1558f, -0.0396f, 0.0029f,
        0.0254f, -0.1074f, 0.9487f, 0.1738f, -0.0439f, 0.0034f,
        0.0264f, -0.1138f, 0.9390f, 0.1929f, -0.0488f, 0.0044f,
        0.0278f, -0.1191f, 0.9282f, 0.2119f, -0.0537f, 0.0049f,
        0.0288f, -0.1245f, 0.9170f, 0.2310f, -0.0581f, 0.0059f,
        0.0293f, -0.1294f, 0.9058f, 0.2510f, -0.0630f, 0.0063f,
        0.0303f, -0.1333f, 0.8926f, 0.2710f, -0.0679f, 0.0073f,
        0.0308f, -0.1367f, 0.8789f, 0.2915f, -0.0728f, 0.0083f,
        0.0308f, -0.1401f, 0.8657f, 0.3120f, -0.0776f, 0.0093f,
        0.0313f, -0.1426f, 0.8506f, 0.3330f, -0.0825f, 0.0103f,
        0.0313f, -0.1445f, 0.8354f, 0.3540f, -0.0874f, 0.0112f,
        0.0313f, -0.1460f, 0.8193f, 0.3755f, -0.0923f, 0.0122f,
        0.0313f, -0.1470f, 0.8022f, 0.3965f, -0.0967f, 0.0137f,
        0.0308f, -0.1479f, 0.7856f, 0.4185f, -0.1016f, 0.0146f,
        0.0303f, -0.1479f, 0.7681f, 0.4399f, -0.1060f, 0.0156f,
        0.0298f, -0.1479f, 0.7505f, 0.4614f, -0.1104f, 0.0166f,
        0.0293f, -0.1470f, 0.7314f, 0.4829f, -0.1147f, 0.0181f,
        0.0288f, -0.1460f, 0.7119f, 0.5049f, -0.1187f, 0.0190f,
        0.0278f, -0.1445f, 0.6929f, 0.5264f, -0.1226f, 0.0200f,
        0.0273f, -0.1431f, 0.6724f, 0.5479f, -0.1260f, 0.0215f,
        0.0264f, -0.1411f, 0.6528f, 0.5693f, -0.1299f, 0.0225f,
        0.0254f, -0.1387f, 0.6323f, 0.5903f, -0.1328f, 0.0234f,
        0.0244f, -0.1357f, 0.6113f, 0.6113f, -0.1357f, 0.0244f,
        0.0234f, -0.1328f, 0.5903f, 0.6323f, -0.1387f, 0.0254f,
        0.0225f, -0.1299f, 0.5693f, 0.6528f, -0.1411f, 0.0264f,
        0.0215f, -0.1260f, 0.5479f, 0.6724f, -0.1431f, 0.0273f,
        0.0200f, -0.1226f, 0.5264f, 0.6929f, -0.1445f, 0.0278f,
        0.0190f, -0.1187f, 0.5049f, 0.7119f, -0.1460f, 0.0288f,
        0.0181f, -0.1147f, 0.4829f, 0.7314f, -0.1470f, 0.0293f,
        0.0166f, -0.1104f, 0.4614f, 0.7505f, -0.1479f, 0.0298f,
        0.0156f, -0.1060f, 0.4399f, 0.7681f, -0.1479f, 0.0303f,
        0.0146f, -0.1016f, 0.4185f, 0.7856f, -0.1479f, 0.0308f,
        0.0137f, -0.0967f, 0.3965f, 0.8022f, -0.1470f, 0.0313f,
        0.0122f, -0.0923f, 0.3755f, 0.8193f, -0.1460f, 0.0313f,
        0.0112f, -0.0874f, 0.3540f, 0.8354f, -0.1445f, 0.0313f,
        0.0103f, -0.0825f, 0.3330f, 0.8506f, -0.1426f, 0.0313f,
        0.0093f, -0.0776f, 0.3120f, 0.8657f, -0.1401f, 0.0308f,
        0.0083f, -0.0728f, 0.2915f, 0.8789f, -0.1367f, 0.0308f,
        0.0073f, -0.0679f, 0.2710f, 0.8926f, -0.1333f, 0.0303f,
        0.0063f, -0.0630f, 0.2510f, 0.9058f, -0.1294f, 0.0293f,
        0.0059f, -0.0581f, 0.2310f, 0.9170f, -0.1245f, 0.0288f,
        0.0049f, -0.0537f, 0.2119f, 0.9282f, -0.1191f, 0.0278f,
        0.0044f, -0.0488f, 0.1929f, 0.9390f, -0.1138f, 0.0264f,
        0.0034f, -0.0439f, 0.1738f, 0.9487f, -0.1074f, 0.0254f,
        0.0029f, -0.0396f, 0.1558f, 0.9575f, -0.1006f, 0.0239f,
        0.0024f, -0.0352f, 0.1382f, 0.9648f, -0.0928f, 0.0225f,
        0.0020f, -0.0308f, 0.1206f, 0.9727f, -0.0850f, 0.0205f,
        0.0015f, -0.0264f, 0.1040f, 0.9785f, -0.0762f, 0.0186f,
        0.0010f, -0.0220f, 0.0874f, 0.9844f, -0.0674f, 0.0166f,
        0.0005f, -0.0181f, 0.0713f, 0.9897f, -0.0576f, 0.0142f,
        0.0005f, -0.0142f, 0.0562f, 0.9932f, -0.0474f, 0.0117f,
        0.0005f, -0.0103f, 0.0415f, 0.9956f, -0.0361f, 0.0088f,
        0.0000f, -0.0068f, 0.0269f, 0.9985f, -0.0249f, 0.0063f,
        0.0000f, -0.0034f, 0.0132f, 1.0000f, -0.0127f, 0.0029f,
    };
    static const float coef_usm_data[384] = {
        0.0000f, -0.6001f, 1.2002f, -0.6001f, 0.0000f, 0.0000f,
        0.0029f, -0.6084f, 1.1987f, -0.5903f, -0.0029f, 0.0000f,
        0.0049f, -0.6147f, 1.1958f, -0.5791f, -0.0068f, 0.0005f,
        0.0073f, -0.6196f, 1.1890f, -0.5659f, -0.0103f, 0.0000f,
        0.0093f, -0.6235f, 1.1802f, -0.5513f, -0.0151f, 0.0000f,
        0.0112f, -0.6265f, 1.1699f, -0.5352f, -0.0195f, 0.0005f,
        0.0122f, -0.6270f, 1.1582f, -0.5181f, -0.0259f, 0.0005f,
        0.0142f, -0.6284f, 1.1455f, -0.5005f, -0.0317f, 0.0005f,
        0.0156f, -0.6265f, 1.1274f, -0.4790f, -0.0386f, 0.0005f,
        0.0166f, -0.6235f, 1.1089f, -0.4570f, -0.0454f, 0.0010f,
        0.0176f, -0.6187f, 1.0879f, -0.4346f, -0.0532f, 0.0010f,
        0.0181f, -0.6138f, 1.0659f, -0.4102f, -0.0615f, 0.0015f,
        0.0190f, -0.6069f, 1.0405f, -0.3843f, -0.0698f, 0.0015f,
        0.0195f, -0.6006f, 1.0161f, -0.3574f, -0.0796f, 0.0020f,
        0.0200f, -0.5928f, 0.9893f, -0.3286f, -0.0898f, 0.0024f,
        0.0200f, -0.5820f, 0.9580f, -0.2988f, -0.1001f, 0.0029f,
        0.0200f, -0.5728f, 0.9292f, -0.2690f, -0.1104f, 0.0034f,
        0.0200f, -0.5620f, 0.8975f, -0.2368f, -0.1226f, 0.0039f,
        0.0205f, -0.5498f, 0.8643f, -0.2046f, -0.1343f, 0.0044f,
        0.0200f, -0.5371f, 0.8301f, -0.1709f, -0.1465f, 0.0049f,
        0.0195f, -0.5239f, 0.7944f, -0.1367f, -0.1587f, 0.0054f,
        0.0195f, -0.5107f, 0.7598f, -0.1021f, -0.1724f, 0.0059f,
        0.0190f, -0.4966f, 0.7231f, -0.0649f, -0.1865f, 0.0063f,
        0.0186f, -0.4819f, 0.6846f, -0.0288f, -0.1997f, 0.0068f,
        0.0186f, -0.4668f, 0.6460f, 0.0093f, -0.2144f, 0.0073f,
        0.0176f, -0.4507f, 0.6055f, 0.0479f, -0.2290f, 0.0083f,
        0.0171f, -0.4370f, 0.5693f, 0.0859f, -0.2446f, 0.0088f,
        0.0161f, -0.4199f, 0.5283f, 0.1255f, -0.2598f, 0.0098f,
        0.0161f, -0.4048f, 0.4883f, 0.1655f, -0.2754f, 0.0103f,
        0.0151f, -0.3887f, 0.4497f, 0.2041f, -0.2910f, 0.0107f,
        0.0142f, -0.3711f, 0.4072f, 0.2446f, -0.3066f, 0.0117f,
        0.0137f, -0.3555f, 0.3672f, 0.2852f, -0.3228f, 0.0122f,
        0.0132f, -0.3394f, 0.3262f, 0.3262f, -0.3394f, 0.0132f,
        0.0122f, -0.3228f, 0.2852f, 0.3672f, -0.3555f, 0.0137f,
        0.0117f, -0.3066f, 0.2446f, 0.4072f, -0.3711f, 0.0142f,
        0.0107f, -0.2910f, 0.2041f, 0.4497f, -0.3887f, 0.0151f,
        0.0103f, -0.2754f, 0.1655f, 0.4883f, -0.4048f, 0.0161f,
        0.0098f, -0.2598f, 0.1255f, 0.5283f, -0.4199f, 0.0161f,
        0.0088f, -0.2446f, 0.0859f, 0.5693f, -0.4370f, 0.0171f,
        0.0083f, -0.2290f, 0.0479f, 0.6055f, -0.4507f, 0.0176f,
        0.0073f, -0.2144f, 0.0093f, 0.6460f, -0.4668f, 0.0186f,
        0.0068f, -0.1997f, -0.0288f, 0.6846f, -0.4819f, 0.0186f,
        0.0063f, -0.1865f, -0.0649f, 0.7231f, -0.4966f, 0.0190f,
        0.0059f, -0.1724f, -0.1021f, 0.7598f, -0.5107f, 0.0195f,
        0.0054f, -0.1587f, -0.1367f, 0.7944f, -0.5239f, 0.0195f,
        0.0049f, -0.1465f, -0.1709f, 0.8301f, -0.5371f, 0.0200f,
        0.0044f, -0.1343f, -0.2046f, 0.8643f, -0.5498f, 0.0205f,
        0.0039f, -0.1226f, -0.2368f, 0.8975f, -0.5620f, 0.0200f,
        0.0034f, -0.1104f, -0.2690f, 0.9292f, -0.5728f, 0.0200f,
        0.0029f, -0.1001f, -0.2988f, 0.9580f, -0.5820f, 0.0200f,
        0.0024f, -0.0898f, -0.3286f, 0.9893f, -0.5928f, 0.0200f,
        0.0020f, -0.0796f, -0.3574f, 1.0161f, -0.6006f, 0.0195f,
        0.0015f, -0.0698f, -0.3843f, 1.0405f, -0.6069f, 0.0190f,
        0.0015f, -0.0615f, -0.4102f, 1.0659f, -0.6138f, 0.0181f,
        0.0010f, -0.0532f, -0.4346f, 1.0879f, -0.6187f, 0.0176f,
        0.0010f, -0.0454f, -0.4570f, 1.1089f, -0.6235f, 0.0166f,
        0.0005f, -0.0386f, -0.4790f, 1.1274f, -0.6265f, 0.0156f,
        0.0005f, -0.0317f, -0.5005f, 1.1455f, -0.6284f, 0.0142f,
        0.0005f, -0.0259f, -0.5181f, 1.1582f, -0.6270f, 0.0122f,
        0.0005f, -0.0195f, -0.5352f, 1.1699f, -0.6265f, 0.0112f,
        0.0000f, -0.0151f, -0.5513f, 1.1802f, -0.6235f, 0.0093f,
        0.0000f, -0.0103f, -0.5659f, 1.1890f, -0.6196f, 0.0073f,
        0.0005f, -0.0068f, -0.5791f, 1.1958f, -0.6147f, 0.0049f,
        0.0000f, -0.0029f, -0.5903f, 1.1987f, -0.6084f, 0.0029f,
    };
	// Upload NIS coefficient tables (program must be active for glUniform)
	gl.UseProgram(program);
	GLint scaleLoc = gl.GetUniformLocation(program, "coef_scale[0]");
	GLint usmLoc = gl.GetUniformLocation(program, "coef_usm[0]");
	if (scaleLoc >= 0)
		gl.Uniform1fv(scaleLoc, 384, coef_scale_data);
	if (usmLoc >= 0)
		gl.Uniform1fv(usmLoc, 384, coef_usm_data);
	gl.UseProgram(0);
}

void NisShader::setTargetScale(const Vec2 &value)
{
	gl.Uniform2f(u_targetScale, value.x, value.y);
}

void NisShader::setSharpness(float value)
{
	gl.Uniform1f(u_sharpness, value);
}
#endif
