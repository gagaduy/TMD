package com.rtm.tamperdetector;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;

public class LocalFileProvider extends ContentProvider {

    @Override
    public boolean onCreate() {
        return true;
    }

    private File getFileForUri(Uri uri) {
        String path = uri.getPath();
        if (path == null) return null;
        if (path.startsWith("/")) path = path.substring(1);
        return new File(getContext().getCacheDir(), path);
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        File file = getFileForUri(uri);
        if (file == null) throw new FileNotFoundException("Invalid URI");
        int imode = ParcelFileDescriptor.MODE_READ_ONLY;
        if (mode.contains("w")) {
            imode = ParcelFileDescriptor.MODE_WRITE_ONLY
                  | ParcelFileDescriptor.MODE_CREATE
                  | ParcelFileDescriptor.MODE_TRUNCATE;
        }
        return ParcelFileDescriptor.open(file, imode);
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder) {
        File file = getFileForUri(uri);
        String[] cols = new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE};
        MatrixCursor cursor = new MatrixCursor(cols, 1);
        if (file != null && file.exists()) {
            cursor.addRow(new Object[]{file.getName(), file.length()});
        }
        return cursor;
    }

    @Override
    public String getType(Uri uri) {
        return "image/jpeg";
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        return null;
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        return 0;
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        return 0;
    }
}
